// iokit_probe2.m — Test doPrepareChainingWithModel: (direct path, bypasses daemon)
// Build: xcrun clang -O2 -fno-objc-arc -o /tmp/iokit_probe2 iokit_probe2.m \
//        -framework Foundation -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t g_tb;
static double tb_us(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e3; }

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║  Direct Chaining Bypass Probe                       ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n\n");

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I   = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR  = NSClassFromString(@"_ANERequest");
        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class g_Client = NSClassFromString(@"_ANEClient");
        Class g_Buffer = NSClassFromString(@"_ANEBuffer");
        Class g_OutputSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class g_ChainingReq = NSClassFromString(@"_ANEChainingRequest");

        int CH = 64, SP = 32;
        int ioBytes = CH * SP * 4;

        // Build dual-procedure model
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func layer0<ios18>(tensor<fp32, [1, %d, 1, %d]> x0) {\n"
            "        string pt0 = const()[name=string(\"pt0\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> st0 = const()[name=string(\"st0\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd0 = const()[name=string(\"pd0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl0 = const()[name=string(\"dl0\"), val=tensor<int32, [2]>([1,1])];\n"
            "        int32 gr0 = const()[name=string(\"gr0\"), val=int32(1)];\n"
            "        string to16_0 = const()[name=string(\"to16_0\"), val=string(\"fp16\")];\n"
            "        tensor<fp16, [1,%d,1,%d]> x16_0 = cast(dtype=to16_0,x=x0)[name=string(\"cin0\")];\n"
            "        tensor<fp16, [%d,%d,1,1]> W0 = const()[name=string(\"W0\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y16_0 = conv(dilations=dl0,groups=gr0,pad=pd0,pad_type=pt0,strides=st0,weight=W0,x=x16_0)"
            "[name=string(\"conv0\")];\n"
            "        string to32_0 = const()[name=string(\"to32_0\"), val=string(\"fp32\")];\n"
            "        tensor<fp32, [1,%d,1,%d]> y0 = cast(dtype=to32_0,x=y16_0)[name=string(\"cout0\")];\n"
            "    } -> (y0);\n"
            "\n"
            "    func layer1<ios18>(tensor<fp32, [1, %d, 1, %d]> x1) {\n"
            "        string pt1 = const()[name=string(\"pt1\"), val=string(\"valid\")];\n"
            "        tensor<int32, [2]> st1 = const()[name=string(\"st1\"), val=tensor<int32, [2]>([1,1])];\n"
            "        tensor<int32, [4]> pd1 = const()[name=string(\"pd1\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "        tensor<int32, [2]> dl1 = const()[name=string(\"dl1\"), val=tensor<int32, [2]>([1,1])];\n"
            "        int32 gr1 = const()[name=string(\"gr1\"), val=int32(1)];\n"
            "        string to16_1 = const()[name=string(\"to16_1\"), val=string(\"fp16\")];\n"
            "        tensor<fp16, [1,%d,1,%d]> x16_1 = cast(dtype=to16_1,x=x1)[name=string(\"cin1\")];\n"
            "        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y16_1 = conv(dilations=dl1,groups=gr1,pad=pd1,pad_type=pt1,strides=st1,weight=W1,x=x16_1)"
            "[name=string(\"conv1\")];\n"
            "        string to32_1 = const()[name=string(\"to32_1\"), val=string(\"fp32\")];\n"
            "        tensor<fp32, [1,%d,1,%d]> y1 = cast(dtype=to32_1,x=y16_1)[name=string(\"cout1\")];\n"
            "    } -> (y1);\n"
            "}\n",
            CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP,
            CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

        // Build identity weights
        int ws = CH * CH * 2;
        int tot = 128 + ws;
        uint8_t *blob = (uint8_t*)calloc(tot, 1);
        blob[0] = 1; blob[4] = 2;
        blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE; blob[68] = 1;
        *(uint32_t*)(blob + 72) = ws;
        *(uint32_t*)(blob + 80) = 128;
        _Float16 *w = (_Float16*)(blob + 128);
        for (int i = 0; i < CH; i++) w[i * CH + i] = (_Float16)1.0f;
        NSData *wdata = [NSData dataWithBytesNoCopy:blob length:tot freeWhenDone:YES];

        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wdata}};

        // Compile
        printf("━━━ Compiling Dual-Procedure Model ━━━\n");
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_D, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_I, @selector(inMemoryModelWithDescriptor:), desc);

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        printf("  Compile: %s\n", ok ? "OK" : "FAIL");
        if (!ok) { printf("  %s\n", [[e description] UTF8String]); return 1; }

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("  Load: %s\n", ok ? "OK" : "FAIL");
        if (!ok) { printf("  %s\n", [[e description] UTF8String]); return 1; }

        // Get _ANEModel
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        printf("  _ANEModel: %s\n", aneModel ? "found" : "nil");

        // Get _ANEClient — both from model and fresh
        id client = ((id(*)(id,SEL,BOOL))objc_msgSend)(
            [g_Client alloc], @selector(initWithRestrictedAccessAllowed:), YES);
        printf("  _ANEClient: %s\n", client ? "acquired" : "nil");

        // Build chaining request
        IOSurfaceRef ioIn = make_surface(ioBytes);
        IOSurfaceRef ioOut = make_surface(ioBytes);

        // Fill input
        IOSurfaceLock(ioIn, 0, NULL);
        float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
        for (int c = 0; c < CH; c++) for (int s = 0; s < SP; s++) inp[c*SP+s] = (float)(s+1) * 0.1f;
        IOSurfaceUnlock(ioIn, 0, NULL);

        id ioObj_in = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_AIO, @selector(objectWithIOSurface:), ioIn);
        id ioObj_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_AIO, @selector(objectWithIOSurface:), ioOut);
        id buf_in = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
            g_Buffer, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
            ioObj_in, @0, (long long)0);  // source 0 = input
        id buf_out = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
            g_Buffer, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
            ioObj_out, @0, (long long)1);  // source 1 = output
        IOSurfaceRef ioStats = make_surface(4096);
        id outputSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
            g_OutputSets, @selector(objectWithstatsSurRef:outputBuffer:), ioStats, @[buf_out]);

        id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
            g_ChainingReq,
            @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
            @[buf_in], @[outputSets], @[], @[], @0, @[], @0, @0, @0);

        printf("\n━━━ Chaining Tests ━━━\n");

        // Test 1: Daemon path (prepareChainingWithModel:) — known to fail
        printf("  Test 1: prepareChainingWithModel (daemon)...\n");
        e = nil;
        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
            aneModel, @{}, cr, 21, &e);
        printf("    Result: %s (err=%ld)\n", ok ? "SUCCESS!!!" : "FAIL", e ? (long)[e code] : 0);

        // Test 2: DIRECT path (doPrepareChainingWithModel:) — NEW, bypasses daemon!
        printf("  Test 2: doPrepareChainingWithModel (DIRECT)...\n");
        e = nil;
        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
            aneModel, @{}, cr, 21, &e);
        printf("    Result: %s", ok ? ">>> SUCCESS!!! CHAINING WORKS! <<<" : "FAIL");
        if (!ok && e) printf(" (code=%ld: %s)", (long)[e code], [[e localizedDescription] UTF8String]);
        printf("\n");

        // Test 3: Try with model's own connection (from _sharedConnection)
        printf("  Test 3: doPrepareChainingWithModel via model's connection...\n");
        {
            unsigned int ic;
            Ivar *ivars = class_copyIvarList([mdl class], &ic);
            id modelClient = nil;
            for (unsigned int i = 0; i < ic; i++) {
                if (strcmp(ivar_getName(ivars[i]), "_sharedConnection") == 0) {
                    modelClient = object_getIvar(mdl, ivars[i]);
                    break;
                }
            }
            free(ivars);

            if (modelClient) {
                printf("    Model's client: <%s>\n", class_getName([modelClient class]));
                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    modelClient, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                    aneModel, @{}, cr, 21, &e);
                printf("    Result: %s", ok ? ">>> SUCCESS!!! <<<" : "FAIL");
                if (!ok && e) printf(" (code=%ld)", (long)[e code]);
                printf("\n");
            } else {
                printf("    No _sharedConnection found\n");
            }
        }

        // Test 4: Try with different options
        printf("  Test 4: doPrepareChainingWithModel with options...\n");
        NSDictionary *opts[] = {
            @{@"enableChaining": @YES},
            @{@"direct": @YES},
            @{@"bypass": @YES},
            @{@"privileged": @YES},
        };
        for (int i = 0; i < 4; i++) {
            e = nil;
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                aneModel, opts[i], cr, 21, &e);
            if (ok) {
                printf("    opts=%s: >>> SUCCESS!!! <<<\n", [[opts[i] description] UTF8String]);
                break;
            } else {
                printf("    opts=%s: FAIL (code=%ld)\n", [[opts[i] description] UTF8String],
                       e ? (long)[e code] : 0);
            }
        }

        // Clean up
        [fm removeItemAtPath:td error:nil];
        CFRelease(ioIn);
        CFRelease(ioOut);
        [mdl release];
        [client release];

        printf("\nDone.\n");
    }
    return 0;
}
