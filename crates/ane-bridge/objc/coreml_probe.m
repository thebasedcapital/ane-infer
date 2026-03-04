// coreml_probe.m — Test CoreML MLProgram compilation path for ANE
// Goal: Compile a model that routes to MLNeuralNetworkEngine (ANE) instead of V1 (CPU)
//
// Build: xcrun clang -O2 -fobjc-arc -o coreml_probe coreml_probe.m \
//        -framework Foundation -framework CoreML -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

static IOSurfaceRef create_iosurface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Recursively walk ObjC object graph looking for _ANEModel
static id find_ane_model(id obj, int depth, NSMutableSet *visited) {
    if (!obj || depth > 8) return nil;
    NSString *addr = [NSString stringWithFormat:@"%p", obj];
    if ([visited containsObject:addr]) return nil;
    [visited addObject:addr];

    const char *cls = class_getName([obj class]);
    if (strcmp(cls, "_ANEModel") == 0) return obj;

    unsigned int count;
    Ivar *ivars = class_copyIvarList([obj class], &count);
    for (unsigned int i = 0; i < count; i++) {
        const char *type = ivar_getTypeEncoding(ivars[i]);
        if (type && type[0] == '@') {
            @try {
                id val = object_getIvar(obj, ivars[i]);
                if (val) {
                    id found = find_ane_model(val, depth + 1, visited);
                    if (found) { free(ivars); return found; }
                }
            } @catch (NSException *ex) { /* skip */ }
        }
    }
    free(ivars);
    return nil;
}

// Build an MLProgram .mlpackage directory structure
static NSURL *build_mlpackage(int CH, int SP) {
    NSString *basePath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"probe_mlpackage"];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm removeItemAtPath:basePath error:nil];

    NSString *pkgPath = [basePath stringByAppendingPathExtension:@"mlpackage"];
    NSString *dataPath = [pkgPath stringByAppendingPathComponent:@"Data"];
    NSString *milPath = [dataPath stringByAppendingPathComponent:@"com.apple.CoreML"];

    [fm createDirectoryAtPath:milPath withIntermediateDirectories:YES attributes:nil error:nil];
    [fm createDirectoryAtPath:[milPath stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];

    // Manifest.json — minimal MLProgram manifest
    NSDictionary *manifest = @{
        @"fileFormatVersion": @"2.0.0",
        @"rootModelIdentifier": @"com.apple.CoreML",
        @"itemInfoEntries": @{
            @"com.apple.CoreML/model.mlmodel": @{
                @"author": @"probe",
                @"description": @"Identity conv for ANE probe",
                @"path": @"com.apple.CoreML/model.mlmodel",
            },
            @"com.apple.CoreML/weights": @{
                @"author": @"probe",
                @"description": @"Weights directory",
                @"path": @"com.apple.CoreML/weights",
                @"isDir": @YES,
            },
        },
    };
    NSData *manifestData = [NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted error:nil];
    [manifestData writeToFile:[pkgPath stringByAppendingPathComponent:@"Manifest.json"] atomically:YES];

    // Build MIL program text
    NSString *mil = [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16,x=x)[name=string(\"cin\")];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
        "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"weights/weight.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y16 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x16)"
        "[name=string(\"conv\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to32,x=y16)[name=string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n", CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

    [mil writeToFile:[milPath stringByAppendingPathComponent:@"model.mil"]
          atomically:YES encoding:NSUTF8StringEncoding error:nil];

    // Build identity weight blob
    int ws = CH * CH * 2;  // FP16 data
    int tot = 128 + ws;    // global header + chunk header + data
    uint8_t *blob = (uint8_t*)calloc(tot, 1);
    blob[0] = 1; blob[4] = 2;  // global header
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE; blob[68] = 1;  // chunk
    *(uint32_t*)(blob + 72) = ws;
    *(uint32_t*)(blob + 80) = 128;
    _Float16 *w = (_Float16*)(blob + 128);
    for (int i = 0; i < CH; i++) w[i * CH + i] = (_Float16)1.0f;

    NSData *wdata = [NSData dataWithBytes:blob length:tot];
    [wdata writeToFile:[milPath stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    free(blob);

    // Also need a model spec .mlmodel protobuf — try without it first
    // CoreML might handle MIL directly in mlpackage format

    printf("  MLPackage at: %s\n", [pkgPath UTF8String]);
    return [NSURL fileURLWithPath:pkgPath];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        int CH = 64, SP = 32;
        int ioBytes = CH * SP * 4;

        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║  CoreML MLProgram → ANE Probe                       ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n\n");

        // Use pre-generated MLProgram .mlpackage (from coremltools Python 3.12)
        NSURL *pkgURL = [NSURL fileURLWithPath:@"/tmp/probe_mlprogram.mlpackage"];
        NSFileManager *fm = [NSFileManager defaultManager];
        if (![fm fileExistsAtPath:[pkgURL path]]) {
            printf("ERROR: /tmp/probe_mlprogram.mlpackage not found!\n");
            printf("Generate with: python3.12 -c 'import coremltools ...'\n");
            return 1;
        }

        // Try compile via CoreML
        printf("━━━ CoreML Compilation (MLProgram) ━━━\n");
        NSError *e = nil;
        NSURL *compiledURL = [MLModel compileModelAtURL:pkgURL error:&e];
        printf("  Compile result: %s\n", compiledURL ? "OK" : "FAIL");
        if (compiledURL) printf("  Compiled to: %s\n", [[compiledURL path] UTF8String]);
        if (!compiledURL && e) printf("  Error: %s\n", [[e description] UTF8String]);

        if (compiledURL) {
            // Load with ALL compute units (forces ANE)
            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsAll;

            // Try E5 runtime flags
            @try {
                if ([config respondsToSelector:@selector(setExperimentalMLE5EngineUsage:)]) {
                    ((void(*)(id,SEL,long))objc_msgSend)(config, @selector(setExperimentalMLE5EngineUsage:), 1);
                    printf("  Set experimentalMLE5EngineUsage = 1\n");
                }
            } @catch (NSException *ex) { /* skip */ }

            e = nil;
            MLModel *model = [MLModel modelWithContentsOfURL:compiledURL configuration:config error:&e];
            printf("  Load result: %s\n", model ? "OK" : "FAIL");
            if (!model && e) printf("  Error: %s\n", [[e description] UTF8String]);

            if (model) {
                // Check what engine it loaded as
                printf("\n━━━ Engine Inspection ━━━\n");
                printf("  Model class: %s\n", class_getName([model class]));

                // Dump ivars
                unsigned int count;
                Ivar *ivars = class_copyIvarList([model class], &count);
                for (unsigned int i = 0; i < count; i++) {
                    const char *name = ivar_getName(ivars[i]);
                    const char *type = ivar_getTypeEncoding(ivars[i]);
                    if (type && type[0] == '@') {
                        @try {
                            id val = object_getIvar(model, ivars[i]);
                            if (val) {
                                printf("  %s = <%s>\n", name, class_getName([val class]));
                                // Check for engine
                                if ([[NSString stringWithUTF8String:class_getName([val class])] containsString:@"Engine"]) {
                                    printf("    >>> ENGINE FOUND: %s <<<\n", class_getName([val class]));
                                    // Check if it's NeuralNetworkEngine (ANE) or V1 (CPU)
                                    if ([val respondsToSelector:@selector(isANEPathForbidden)]) {
                                        BOOL forbidden = ((BOOL(*)(id,SEL))objc_msgSend)(val, @selector(isANEPathForbidden));
                                        printf("    isANEPathForbidden: %s\n", forbidden ? "YES" : "NO");
                                    }
                                }
                            }
                        } @catch (NSException *ex) { /* skip */ }
                    }
                }
                free(ivars);

                // Dig into engine internals
                printf("\n━━━ Engine Internals ━━━\n");
                id engine = nil;
                @try {
                    unsigned int mc;
                    Ivar *mivars = class_copyIvarList([model class], &mc);
                    for (unsigned int mi = 0; mi < mc; mi++) {
                        if (strcmp(ivar_getName(mivars[mi]), "_internalEngine") == 0) {
                            engine = object_getIvar(model, mivars[mi]);
                            break;
                        }
                    }
                    free(mivars);
                } @catch (NSException *ex) { /* skip */ }
                if (engine) {
                    unsigned int ecount;
                    Ivar *eivars = class_copyIvarList([engine class], &ecount);
                    for (unsigned int ei = 0; ei < ecount; ei++) {
                        const char *ename = ivar_getName(eivars[ei]);
                        const char *etype = ivar_getTypeEncoding(eivars[ei]);
                        printf("  %s (%s)", ename, etype ?: "?");
                        if (etype && etype[0] == '@') {
                            @try {
                                id val = object_getIvar(engine, eivars[ei]);
                                if (val) {
                                    printf(" = <%s>", class_getName([val class]));
                                    // Dig one more level for ANE-related classes
                                    const char *vcls = class_getName([val class]);
                                    if (strstr(vcls, "ANE") || strstr(vcls, "Neural") ||
                                        strstr(vcls, "Compiled") || strstr(vcls, "Program") ||
                                        strstr(vcls, "Delegate")) {
                                        unsigned int vcount;
                                        Ivar *vivars = class_copyIvarList([val class], &vcount);
                                        for (unsigned int vi = 0; vi < vcount; vi++) {
                                            const char *vname = ivar_getName(vivars[vi]);
                                            const char *vtype = ivar_getTypeEncoding(vivars[vi]);
                                            if (vtype && vtype[0] == '@') {
                                                @try {
                                                    id vval = object_getIvar(val, vivars[vi]);
                                                    if (vval) printf("\n      .%s = <%s>", vname, class_getName([vval class]));
                                                } @catch (NSException *ex) { /* skip */ }
                                            }
                                        }
                                        free(vivars);
                                    }
                                }
                            } @catch (NSException *ex) { /* skip */ }
                        }
                        printf("\n");
                    }
                    free(eivars);
                }

                // Dig into _functionNameToEngineMap
                printf("\n━━━ Function→Engine Map ━━━\n");
                @try {
                    unsigned int ec;
                    Ivar *eivars2 = class_copyIvarList([engine class], &ec);
                    for (unsigned int ei = 0; ei < ec; ei++) {
                        if (strcmp(ivar_getName(eivars2[ei]), "_functionNameToEngineMap") == 0) {
                            NSDictionary *map = object_getIvar(engine, eivars2[ei]);
                            for (NSString *key in map) {
                                id subEngine = map[key];
                                printf("  '%s' → <%s>\n", [key UTF8String], class_getName([subEngine class]));
                                // Dump ALL sub-engine ivars
                                // First check superclass chain
                                printf("    Superclass chain:");
                                Class cls = [subEngine class];
                                while (cls) {
                                    printf(" %s →", class_getName(cls));
                                    cls = class_getSuperclass(cls);
                                }
                                printf(" nil\n");

                                // Dump ivars from ALL classes in hierarchy
                                Class walkCls = [subEngine class];
                                while (walkCls && walkCls != [NSObject class]) {
                                    unsigned int wc;
                                    Ivar *wivars = class_copyIvarList(walkCls, &wc);
                                    if (wc > 0) {
                                        printf("    --- %s (%u ivars) ---\n", class_getName(walkCls), wc);
                                        for (unsigned int wi = 0; wi < wc; wi++) {
                                            const char *wn = ivar_getName(wivars[wi]);
                                            const char *wt = ivar_getTypeEncoding(wivars[wi]);
                                            printf("      %s (%s)", wn, wt ?: "?");
                                            if (wt && wt[0] == '@') {
                                                @try {
                                                    id wv = object_getIvar(subEngine, wivars[wi]);
                                                    if (wv) printf(" = <%s>", class_getName([wv class]));
                                                } @catch (NSException *ex) { /* skip */ }
                                            }
                                            printf("\n");
                                        }
                                    }
                                    free(wivars);
                                    walkCls = class_getSuperclass(walkCls);
                                }

                                unsigned int sc;
                                Ivar *sivars = class_copyIvarList([subEngine class], &sc);
                                for (unsigned int si = 0; si < sc; si++) {
                                    const char *sn = ivar_getName(sivars[si]);
                                    const char *st = ivar_getTypeEncoding(sivars[si]);
                                    printf("    .%s", sn);
                                    if (st && st[0] == '@') {
                                        @try {
                                            id sv = object_getIvar(subEngine, sivars[si]);
                                            if (sv) {
                                                printf(" = <%s>", class_getName([sv class]));
                                                // Level 2 dump for interesting classes
                                                unsigned int vc;
                                                Ivar *vivars = class_copyIvarList([sv class], &vc);
                                                for (unsigned int vi = 0; vi < vc; vi++) {
                                                    const char *vn = ivar_getName(vivars[vi]);
                                                    const char *vt = ivar_getTypeEncoding(vivars[vi]);
                                                    if (vt && vt[0] == '@') {
                                                        @try {
                                                            id vv = object_getIvar(sv, vivars[vi]);
                                                            if (vv) {
                                                                printf("\n        .%s = <%s>", vn, class_getName([vv class]));
                                                            }
                                                        } @catch (NSException *ex) { /* skip */ }
                                                    }
                                                }
                                                free(vivars);
                                            }
                                        } @catch (NSException *ex) { /* skip */ }
                                    }
                                    printf("\n");
                                }
                                free(sivars);
                            }
                            break;
                        }
                    }
                    free(eivars2);
                } @catch (NSException *ex) {
                    printf("  Error: %s\n", [[ex reason] UTF8String]);
                }

                // Check ANE flags on the engine
                printf("\n━━━ ANE Path Check ━━━\n");
                @try {
                    // Get the MLProgramEngine from the function map
                    unsigned int ec2;
                    Ivar *eivars3 = class_copyIvarList([engine class], &ec2);
                    id progEngine = nil;
                    for (unsigned int ei = 0; ei < ec2; ei++) {
                        if (strcmp(ivar_getName(eivars3[ei]), "_functionNameToEngineMap") == 0) {
                            NSDictionary *map = object_getIvar(engine, eivars3[ei]);
                            progEngine = map[@"main"];
                            break;
                        }
                    }
                    free(eivars3);

                    if (progEngine) {
                        // Read boolean ivars from MLNeuralNetworkEngine
                        if ([progEngine respondsToSelector:@selector(isANEPathForbidden)]) {
                            BOOL forbidden = ((BOOL(*)(id,SEL))objc_msgSend)(progEngine, @selector(isANEPathForbidden));
                            printf("  isANEPathForbidden: %s\n", forbidden ? "YES (CPU only)" : "NO (ANE enabled!)");
                        }
                        if ([progEngine respondsToSelector:@selector(isGPUPathForbidden)]) {
                            BOOL gpuForbidden = ((BOOL(*)(id,SEL))objc_msgSend)(progEngine, @selector(isGPUPathForbidden));
                            printf("  isGPUPathForbidden: %s\n", gpuForbidden ? "YES" : "NO");
                        }
                        if ([progEngine respondsToSelector:@selector(modelIsMIL)]) {
                            BOOL isMIL = ((BOOL(*)(id,SEL))objc_msgSend)(progEngine, @selector(modelIsMIL));
                            printf("  modelIsMIL: %s\n", isMIL ? "YES" : "NO");
                        }
                        if ([progEngine respondsToSelector:@selector(usingCPU)]) {
                            BOOL cpu = ((BOOL(*)(id,SEL))objc_msgSend)(progEngine, @selector(usingCPU));
                            printf("  usingCPU: %s\n", cpu ? "YES" : "NO");
                        }

                        // Check _compilerOutput for ANE model
                        Class nneCls = NSClassFromString(@"MLNeuralNetworkEngine");
                        unsigned int nc;
                        Ivar *nivars = class_copyIvarList(nneCls, &nc);
                        for (unsigned int ni = 0; ni < nc; ni++) {
                            if (strcmp(ivar_getName(nivars[ni]), "_compilerOutput") == 0) {
                                id compOut = object_getIvar(progEngine, nivars[ni]);
                                if (compOut) {
                                    printf("  _compilerOutput = <%s>\n", class_getName([compOut class]));
                                    unsigned int cc;
                                    Ivar *civars = class_copyIvarList([compOut class], &cc);
                                    for (unsigned int ci = 0; ci < cc; ci++) {
                                        const char *cn = ivar_getName(civars[ci]);
                                        const char *ct = ivar_getTypeEncoding(civars[ci]);
                                        if (ct && ct[0] == '@') {
                                            @try {
                                                id cv = object_getIvar(compOut, civars[ci]);
                                                if (cv) printf("    .%s = <%s>\n", cn, class_getName([cv class]));
                                            } @catch (NSException *ex) { /* skip */ }
                                        }
                                    }
                                    free(civars);
                                } else {
                                    printf("  _compilerOutput = nil\n");
                                }
                                break;
                            }
                        }
                        free(nivars);
                    }
                } @catch (NSException *ex) {
                    printf("  Error: %s\n", [[ex reason] UTF8String]);
                }

                // Deep search for _ANEModel
                printf("\n━━━ Deep Walk for _ANEModel ━━━\n");
                NSMutableSet *visited = [NSMutableSet set];
                id aneModel = find_ane_model(model, 0, visited);
                printf("  Result: %s\n", aneModel ? "FOUND _ANEModel!" : "not found");

                if (aneModel) {
                    printf("\n━━━ Chaining Test ━━━\n");
                    // Try chaining with this CoreML-compiled _ANEModel
                    Class g_ANEClient = NSClassFromString(@"_ANEClient");
                    id client = ((id(*)(id,SEL,BOOL))objc_msgSend)(
                        [g_ANEClient alloc], @selector(initWithRestrictedAccessAllowed:), YES);

                    if (client) {
                        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
                        Class g_Buffer = NSClassFromString(@"_ANEBuffer");
                        Class g_OutputSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
                        Class g_ChainingReq = NSClassFromString(@"_ANEChainingRequest");

                        IOSurfaceRef ioIn = create_iosurface(ioBytes);
                        IOSurfaceRef ioOut = create_iosurface(ioBytes);

                        id ioObj_in = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                        id ioObj_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                        id buf_in = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(g_Buffer,
                            @selector(bufferWithIOSurfaceObject:source:symbolIndex:), ioObj_in, @0, (long long)0);
                        id buf_out = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(g_Buffer,
                            @selector(bufferWithIOSurfaceObject:source:symbolIndex:), ioObj_out, @1, (long long)0);
                        id outputSets = ((id(*)(Class,SEL,id))objc_msgSend)(g_OutputSets,
                            @selector(outputSetsWithBuffers:), @[buf_out]);

                        id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(g_ChainingReq,
                            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                            @[buf_in], @[outputSets], @[], @[], @0, @[], @0, @0, @0);

                        printf("  ChainingRequest created: %s\n", cr ? "YES" : "NO");
                        if (cr) {
                            BOOL validates = ((BOOL(*)(id,SEL))objc_msgSend)(cr, @selector(validate));
                            printf("  Validates: %s\n", validates ? "YES" : "NO");
                        }

                        // Try prepareChainingWithModel
                        e = nil;
                        BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                            aneModel, @{}, cr, 21, &e);
                        printf("  prepareChainingWithModel: %s\n", ok ? "SUCCESS !!!" : "FAIL");
                        if (!ok && e) printf("  Error: %s (code=%ld)\n", [[e localizedDescription] UTF8String], (long)[e code]);

                        CFRelease(ioIn);
                        CFRelease(ioOut);
                    }
                }
            }
        }

        printf("\nDone.\n");
    }
    return 0;
}
