// iokit_probe.m — Direct IOKit access to Apple Neural Engine
// Bypass the entire AppleNeuralEngine.framework, talk directly to the kernel driver.
//
// Build: xcrun clang -O2 -fobjc-arc -o /tmp/iokit_probe iokit_probe.m \
//        -framework Foundation -framework IOKit -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach.h>

static void dump_properties(io_registry_entry_t entry, int depth) {
    CFMutableDictionaryRef props = NULL;
    kern_return_t kr = IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0);
    if (kr == KERN_SUCCESS && props) {
        NSString *desc = [(__bridge NSDictionary *)props description];
        // Truncate long output
        if ([desc length] > 2000) desc = [[desc substringToIndex:2000] stringByAppendingString:@"..."];
        printf("%*s Properties: %s\n", depth*2, "", [desc UTF8String]);
        CFRelease(props);
    }
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);

        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║  IOKit Direct ANE Probe                             ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n\n");

        // ═══════════════════════════════════════════════════════
        // Step 1: Find ANE IOService
        // ═══════════════════════════════════════════════════════
        printf("━━━ Step 1: Find ANE IOService ━━━\n");

        // Try multiple matching patterns
        const char *serviceNames[] = {
            "AppleH13ANEDevice",   // M1
            "AppleH14ANEDevice",   // M2
            "AppleH15ANEDevice",   // M3
            "AppleH16ANEDevice",   // M4
            "AppleH17ANEDevice",   // M5
            "AppleANEDevice",
            "AppleNeuralEngine",
            "H13ANEIn",
            "ANEDevice",
        };

        io_service_t aneService = IO_OBJECT_NULL;
        for (int i = 0; i < sizeof(serviceNames)/sizeof(serviceNames[0]); i++) {
            CFMutableDictionaryRef matching = IOServiceMatching(serviceNames[i]);
            io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, matching);
            if (svc != IO_OBJECT_NULL) {
                printf("  Found: %s (service=0x%x)\n", serviceNames[i], svc);
                aneService = svc;
                break;
            }
        }

        // Also try name matching
        if (aneService == IO_OBJECT_NULL) {
            printf("  No direct match. Searching IOService tree...\n");
            io_iterator_t iter;
            kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault,
                IOServiceMatching("IOService"), &iter);
            if (kr == KERN_SUCCESS) {
                io_service_t svc;
                while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
                    io_name_t name;
                    IORegistryEntryGetName(svc, name);
                    if (strstr(name, "ANE") || strstr(name, "ane") ||
                        strstr(name, "Neural") || strstr(name, "neural")) {
                        printf("  Found in tree: %s (service=0x%x)\n", name, svc);
                        if (aneService == IO_OBJECT_NULL) aneService = svc;
                        else IOObjectRelease(svc);
                    } else {
                        IOObjectRelease(svc);
                    }
                }
                IOObjectRelease(iter);
            }
        }

        // Search by class name
        if (aneService == IO_OBJECT_NULL) {
            printf("  Trying class-based search...\n");
            io_iterator_t iter;
            kern_return_t kr = IOServiceGetMatchingServices(kIOMainPortDefault,
                IOServiceMatching("IOUserClient"), &iter);
            if (kr == KERN_SUCCESS) {
                io_service_t svc;
                int checked = 0;
                while ((svc = IOIteratorNext(iter)) != IO_OBJECT_NULL && checked < 500) {
                    io_name_t name, className;
                    IORegistryEntryGetName(svc, name);
                    IOObjectGetClass(svc, className);
                    if (strstr(name, "ANE") || strstr(className, "ANE") ||
                        strstr(name, "Neural") || strstr(className, "Neural")) {
                        printf("  Found client: %s (class: %s, 0x%x)\n", name, className, svc);
                    }
                    IOObjectRelease(svc);
                    checked++;
                }
                IOObjectRelease(iter);
            }
        }

        if (aneService == IO_OBJECT_NULL) {
            // Try searching all registry for anything ANE-related
            printf("\n  Searching full IORegistry for ANE...\n");
            io_iterator_t iter;
            kern_return_t kr = IORegistryEntryCreateIterator(
                IORegistryGetRootEntry(kIOMainPortDefault),
                kIOServicePlane, kIORegistryIterateRecursively, &iter);
            if (kr == KERN_SUCCESS) {
                io_registry_entry_t entry;
                int found = 0;
                while ((entry = IOIteratorNext(iter)) != IO_OBJECT_NULL && found < 10) {
                    io_name_t name, className;
                    IORegistryEntryGetName(entry, name);
                    IOObjectGetClass(entry, className);
                    if (strstr(name, "ANE") || strstr(name, "ane") ||
                        strstr(className, "ANE") || strstr(className, "ane") ||
                        strstr(name, "neural") || strstr(name, "Neural")) {
                        printf("  Registry entry: %s (class: %s)\n", name, className);
                        dump_properties(entry, 2);
                        if (aneService == IO_OBJECT_NULL &&
                            !strstr(className, "UserClient")) {
                            aneService = entry;
                            IOObjectRetain(aneService);
                        }
                        found++;
                    }
                    IOObjectRelease(entry);
                }
                IOObjectRelease(iter);
                printf("  Scanned registry, found %d ANE entries\n", found);
            }
        }

        // Try opening each ANE service
        printf("\n━━━ Trying All ANE Services ━━━\n");
        {
            io_iterator_t iter;
            kern_return_t kr2 = IORegistryEntryCreateIterator(
                IORegistryGetRootEntry(kIOMainPortDefault),
                kIOServicePlane, kIORegistryIterateRecursively, &iter);
            if (kr2 == KERN_SUCCESS) {
                io_registry_entry_t entry;
                while ((entry = IOIteratorNext(iter)) != IO_OBJECT_NULL) {
                    io_name_t name, className;
                    IORegistryEntryGetName(entry, name);
                    IOObjectGetClass(entry, className);
                    if (strstr(name, "ANE") || strstr(name, "ane") ||
                        strstr(className, "ANE") || strstr(className, "ane") ||
                        strstr(name, "H11ANE")) {
                        for (uint32_t type = 0; type <= 5; type++) {
                            io_connect_t conn = IO_OBJECT_NULL;
                            kern_return_t okr = IOServiceOpen(entry, mach_task_self(), type, &conn);
                            if (okr == KERN_SUCCESS) {
                                printf("  %s (%s) type=%u: OPENED! (conn=0x%x)\n",
                                       name, className, type, conn);
                                // Probe ALL method types for first 32 selectors
                                for (uint32_t sel = 0; sel < 32; sel++) {
                                    // Try scalar method
                                    {
                                        uint64_t out[4] = {0};
                                        uint32_t oc = 4;
                                        kern_return_t mr = IOConnectCallScalarMethod(conn, sel, NULL, 0, out, &oc);
                                        if (mr == KERN_SUCCESS) {
                                            printf("    sel %2u scalar: OK outputs=[", sel);
                                            for (uint32_t j = 0; j < oc; j++) printf("%s0x%llx", j?",":"", out[j]);
                                            printf("]\n");
                                        } else if (mr != 0xe00002c2 && mr != 0xe00002c7 && mr != KERN_FAILURE) {
                                            printf("    sel %2u scalar: 0x%x (%s)\n", sel, mr, mach_error_string(mr));
                                        }
                                    }
                                    // Try struct method
                                    {
                                        char out[4096] = {0};
                                        size_t outSize = sizeof(out);
                                        kern_return_t mr = IOConnectCallStructMethod(conn, sel, NULL, 0, out, &outSize);
                                        if (mr == KERN_SUCCESS) {
                                            printf("    sel %2u struct: OK (%zu bytes)\n", sel, outSize);
                                        } else if (mr != 0xe00002c2 && mr != 0xe00002c7 && mr != KERN_FAILURE) {
                                            printf("    sel %2u struct: 0x%x (%s)\n", sel, mr, mach_error_string(mr));
                                        }
                                    }
                                }
                                IOServiceClose(conn);
                            }
                        }
                    }
                    IOObjectRelease(entry);
                }
                IOObjectRelease(iter);
            }
        }

        if (aneService == IO_OBJECT_NULL) {
            printf("\n  ERROR: No ANE IOService found!\n");
            printf("  The ANE kernel driver may use a different naming convention.\n");

            // Check what the framework connects to internally
            printf("\n━━━ Checking Framework Connection ━━━\n");
            void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
            if (handle) {
                Class clientCls = NSClassFromString(@"_ANEClient");
                if (clientCls) {
                    id client = ((id(*)(id,SEL,BOOL))objc_msgSend)(
                        [clientCls alloc], @selector(initWithRestrictedAccessAllowed:), YES);
                    if (client) {
                        printf("  _ANEClient acquired.\n");
                        // Check for internal connection info
                        unsigned int count;
                        Ivar *ivars = class_copyIvarList(clientCls, &count);
                        for (unsigned int i = 0; i < count; i++) {
                            const char *n = ivar_getName(ivars[i]);
                            const char *t = ivar_getTypeEncoding(ivars[i]);
                            printf("    .%s (%s)", n, t ?: "?");
                            if (t && t[0] == '@') {
                                @try {
                                    id v = object_getIvar(client, ivars[i]);
                                    if (v) printf(" = <%s>", class_getName([v class]));
                                } @catch (NSException *ex) { /* */ }
                            }
                            printf("\n");
                        }
                        free(ivars);

                        // Check _ANEDaemonConnection
                        Class dcCls = NSClassFromString(@"_ANEDaemonConnection");
                        if (dcCls) {
                            printf("\n  _ANEDaemonConnection ivars:\n");
                            Ivar *divars = class_copyIvarList(dcCls, &count);
                            for (unsigned int i = 0; i < count; i++) {
                                const char *n = ivar_getName(divars[i]);
                                const char *t = ivar_getTypeEncoding(divars[i]);
                                printf("    .%s (%s)\n", n, t ?: "?");
                            }
                            free(divars);
                        }

                        // Check _ANEDeviceController
                        Class devCtrl = NSClassFromString(@"_ANEDeviceController");
                        if (devCtrl) {
                            printf("\n  _ANEDeviceController ivars:\n");
                            Ivar *dcivars = class_copyIvarList(devCtrl, &count);
                            for (unsigned int i = 0; i < count; i++) {
                                const char *n = ivar_getName(dcivars[i]);
                                const char *t = ivar_getTypeEncoding(dcivars[i]);
                                printf("    .%s (%s)\n", n, t ?: "?");
                            }
                            free(dcivars);

                            // Try to get a device controller instance
                            if ([devCtrl respondsToSelector:@selector(sharedController)]) {
                                id ctrl = ((id(*)(Class,SEL))objc_msgSend)(devCtrl, @selector(sharedController));
                                if (ctrl) {
                                    printf("    sharedController: <%s>\n", class_getName([ctrl class]));
                                    // Dump its ivars with values
                                    Ivar *civars = class_copyIvarList([ctrl class], &count);
                                    for (unsigned int i = 0; i < count; i++) {
                                        const char *n = ivar_getName(civars[i]);
                                        const char *t = ivar_getTypeEncoding(civars[i]);
                                        if (t && t[0] == '@') {
                                            @try {
                                                id v = object_getIvar(ctrl, civars[i]);
                                                if (v) printf("      .%s = <%s>\n", n, class_getName([v class]));
                                            } @catch (NSException *ex) { /* */ }
                                        } else if (t && strcmp(t, "I") == 0) {
                                            // Try reading uint
                                            @try {
                                                ptrdiff_t off = ivar_getOffset(civars[i]);
                                                uint32_t val = *(uint32_t*)((char*)(__bridge void*)ctrl + off);
                                                printf("      .%s = %u\n", n, val);
                                            } @catch (NSException *ex) { /* */ }
                                        }
                                    }
                                    free(civars);
                                }
                            }
                        }
                    }
                }
            }

            printf("\nDone.\n");
            return 0;
        }

        // ═══════════════════════════════════════════════════════
        // Step 2: Open user client
        // ═══════════════════════════════════════════════════════
        printf("\n━━━ Step 2: Open User Client ━━━\n");
        io_connect_t connect = IO_OBJECT_NULL;
        kern_return_t kr = IOServiceOpen(aneService, mach_task_self(), 0, &connect);
        printf("  IOServiceOpen(type=0): %s (0x%x)\n",
               kr == KERN_SUCCESS ? "OK" : mach_error_string(kr), kr);

        if (kr != KERN_SUCCESS) {
            // Try different types
            for (uint32_t type = 1; type <= 10; type++) {
                kr = IOServiceOpen(aneService, mach_task_self(), type, &connect);
                if (kr == KERN_SUCCESS) {
                    printf("  IOServiceOpen(type=%u): OK!\n", type);
                    break;
                }
            }
        }

        if (connect == IO_OBJECT_NULL) {
            printf("  Could not open user client.\n");
        } else {
            // ═══════════════════════════════════════════════════════
            // Step 3: Probe user client methods
            // ═══════════════════════════════════════════════════════
            printf("\n━━━ Step 3: Probe User Client Methods ━━━\n");
            for (uint32_t sel = 0; sel < 32; sel++) {
                uint64_t output[4] = {0};
                uint32_t outputCount = 4;
                kr = IOConnectCallScalarMethod(connect, sel, NULL, 0, output, &outputCount);
                if (kr != KERN_FAILURE && kr != 0xe00002c2) { // Skip generic failures
                    printf("  Selector %2u: %s (0x%x)", sel, mach_error_string(kr), kr);
                    if (kr == KERN_SUCCESS && outputCount > 0) {
                        printf(" outputs=[");
                        for (uint32_t i = 0; i < outputCount; i++)
                            printf("%s0x%llx", i ? "," : "", output[i]);
                        printf("]");
                    }
                    printf("\n");
                }
            }

            IOServiceClose(connect);
        }

        // ═══════════════════════════════════════════════════════
        // Step 4: Check framework's internal IOKit port
        // ═══════════════════════════════════════════════════════
        printf("\n━━━ Step 4: Framework IOKit Port ━━━\n");
        {
            void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
            if (handle) {
                Class devCtrl = NSClassFromString(@"_ANEDeviceController");
                if (devCtrl) {
                    // Check methods
                    unsigned int mcount;
                    Method *methods = class_copyMethodList(devCtrl, &mcount);
                    printf("  _ANEDeviceController methods (%u):\n", mcount);
                    for (unsigned int i = 0; i < mcount; i++) {
                        SEL sel = method_getName(methods[i]);
                        const char *types = method_getTypeEncoding(methods[i]);
                        printf("    %s (%s)\n", sel_getName(sel), types ?: "?");
                    }
                    free(methods);

                    // Try sharedController
                    id ctrl = nil;
                    if ([devCtrl respondsToSelector:@selector(sharedController)]) {
                        ctrl = ((id(*)(Class,SEL))objc_msgSend)(devCtrl, @selector(sharedController));
                    }
                    if (!ctrl && [devCtrl respondsToSelector:@selector(new)]) {
                        ctrl = ((id(*)(Class,SEL))objc_msgSend)(devCtrl, @selector(new));
                    }
                    if (ctrl) {
                        printf("\n  Got controller: <%s>\n", class_getName([ctrl class]));
                        unsigned int ic;
                        Ivar *ivars = class_copyIvarList([ctrl class], &ic);
                        for (unsigned int i = 0; i < ic; i++) {
                            const char *n = ivar_getName(ivars[i]);
                            const char *t = ivar_getTypeEncoding(ivars[i]);
                            printf("    .%s (%s)", n, t ?: "?");
                            if (t && t[0] == '@') {
                                @try {
                                    id v = object_getIvar(ctrl, ivars[i]);
                                    if (v) printf(" = <%s>", class_getName([v class]));
                                } @catch (NSException *ex) { /* */ }
                            } else if (t && t[0] == 'I') {
                                ptrdiff_t off = ivar_getOffset(ivars[i]);
                                uint32_t val = *(uint32_t*)((char*)(__bridge void*)ctrl + off);
                                printf(" = %u (0x%x)", val, val);
                            }
                            printf("\n");
                        }
                        free(ivars);
                    }
                }

                // Get _ANEClient and dump its device connection
                Class clientCls2 = NSClassFromString(@"_ANEClient");
                if (clientCls2) {
                    id client = ((id(*)(id,SEL,BOOL))objc_msgSend)(
                        [clientCls2 alloc], @selector(initWithRestrictedAccessAllowed:), YES);
                    if (client) {
                        printf("\n  _ANEClient methods:\n");
                        unsigned int mc;
                        Method *ms = class_copyMethodList(clientCls2, &mc);
                        for (unsigned int i = 0; i < mc; i++) {
                            SEL sel = method_getName(ms[i]);
                            const char *name = sel_getName(sel);
                            // Only show interesting ones
                            if (strstr(name, "device") || strstr(name, "Device") ||
                                strstr(name, "connect") || strstr(name, "Connect") ||
                                strstr(name, "port") || strstr(name, "Port") ||
                                strstr(name, "iokit") || strstr(name, "IOKit") ||
                                strstr(name, "direct") || strstr(name, "Direct") ||
                                strstr(name, "dispatch") || strstr(name, "Dispatch") ||
                                strstr(name, "evaluate") || strstr(name, "Evaluate") ||
                                strstr(name, "chain") || strstr(name, "Chain") ||
                                strstr(name, "compile") || strstr(name, "Compile") ||
                                strstr(name, "load") || strstr(name, "Load") ||
                                strstr(name, "prepare") || strstr(name, "Prepare")) {
                                printf("    %s (%s)\n", name, method_getTypeEncoding(ms[i]) ?: "?");
                            }
                        }
                        free(ms);
                    }
                }

                // Check _ANEDeviceStruct
                Class devStruct = NSClassFromString(@"_ANEDeviceStruct");
                if (devStruct) {
                    printf("\n  _ANEDeviceStruct ivars:\n");
                    unsigned int ic;
                    Ivar *ivars = class_copyIvarList(devStruct, &ic);
                    for (unsigned int i = 0; i < ic; i++) {
                        printf("    .%s (%s)\n", ivar_getName(ivars[i]),
                               ivar_getTypeEncoding(ivars[i]) ?: "?");
                    }
                    free(ivars);
                }
            }
        }

        if (aneService != IO_OBJECT_NULL) IOObjectRelease(aneService);
        printf("\nDone.\n");
    }
    return 0;
}
