// chaining_e2e.m — End-to-end ANE chaining: prepare → buffersReady → enqueueSets → verify output
// Build: xcrun clang -O2 -fno-objc-arc -o /tmp/chaining_e2e chaining_e2e.m \
//        -framework Foundation -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#include <math.h>

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
        printf("║  ANE Chaining End-to-End Test                       ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n\n");

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class g_Client = NSClassFromString(@"_ANEClient");
        Class g_Buffer = NSClassFromString(@"_ANEBuffer");
        Class g_OutputSets = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class g_ChainingReq = NSClassFromString(@"_ANEChainingRequest");
        Class g_InputReady = NSClassFromString(@"_ANEInputBuffersReady");
        Class g_OutputEnq = NSClassFromString(@"_ANEOutputSetEnqueue");

        int CH = 64, SP = 32;
        int ioBytes = CH * SP * 4;

        // ── Build DUAL-procedure identity conv model (council says chaining needs ≥2 procs) ──
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
            // layer0
            CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP,
            // layer1
            CH, SP, CH, SP, CH, CH, CH, CH, CH, SP, CH, SP];

        // Identity weights
        int ws = CH * CH * 2;
        int tot = 128 + ws;
        uint8_t *wblob = (uint8_t*)calloc(tot, 1);
        wblob[0] = 1; wblob[4] = 2;
        wblob[64] = 0xEF; wblob[65] = 0xBE; wblob[66] = 0xAD; wblob[67] = 0xDE; wblob[68] = 1;
        *(uint32_t*)(wblob + 72) = ws;
        *(uint32_t*)(wblob + 80) = 128;
        _Float16 *wf = (_Float16*)(wblob + 128);
        for (int i = 0; i < CH; i++) wf[i * CH + i] = (_Float16)1.0f;
        NSData *wdata = [NSData dataWithBytesNoCopy:wblob length:tot freeWhenDone:YES];
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wdata}};

        // Compile + Load
        printf("━━━ Compile & Load ━━━\n");
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [wdata writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        printf("  Model loaded, _ANEModel: %s\n", aneModel ? "OK" : "nil");

        // Get client
        id client = ((id(*)(id,SEL,BOOL))objc_msgSend)([g_Client alloc], @selector(initWithRestrictedAccessAllowed:), YES);

        // ── First: verify normal eval works ──
        printf("\n━━━ Baseline: Normal Eval ━━━\n");
        {
            IOSurfaceRef ioIn = make_surface(ioBytes);
            IOSurfaceRef ioOut = make_surface(ioBytes);
            IOSurfaceLock(ioIn, 0, NULL);
            float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
            for (int c = 0; c < CH; c++) for (int s = 0; s < SP; s++) inp[c*SP+s] = (c == 0) ? 1.0f : 0.0f;
            IOSurfaceUnlock(ioIn, 0, NULL);

            id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
            id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                mdl, @{}, req, 21, &e);
            printf("  Normal eval: %s\n", ok ? "OK" : "FAIL");

            IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
            float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
            printf("  Output[0..3] = [%.4f, %.4f, %.4f, %.4f]\n", out[0], out[1], out[2], out[3]);
            IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
            CFRelease(ioIn); CFRelease(ioOut);
        }

        // ── Chaining: prepare → buffersReady → enqueueSets ──
        printf("\n━━━ Chaining: Full Sequence ━━━\n");
        {
            IOSurfaceRef ioIn = make_surface(ioBytes);
            IOSurfaceRef ioOut = make_surface(ioBytes);
            IOSurfaceRef ioStats = make_surface(4096);

            // Fill input: channel 0 = 1.0, rest = 0.0 (identity conv should pass through)
            IOSurfaceLock(ioIn, 0, NULL);
            float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
            for (int c = 0; c < CH; c++) for (int s = 0; s < SP; s++) inp[c*SP+s] = (c == 0) ? 1.0f : 0.0f;
            IOSurfaceUnlock(ioIn, 0, NULL);

            id ioObj_in = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
            id ioObj_out = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);

            id buf_in = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(g_Buffer,
                @selector(bufferWithIOSurfaceObject:symbolIndex:source:), ioObj_in, @0, (long long)0);
            id buf_out = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(g_Buffer,
                @selector(bufferWithIOSurfaceObject:symbolIndex:source:), ioObj_out, @0, (long long)1);

            id outputSets = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(g_OutputSets,
                @selector(objectWithstatsSurRef:outputBuffer:), ioStats, @[buf_out]);

            // Build second output set for procedure 1
            IOSurfaceRef ioOut2 = make_surface(ioBytes);
            IOSurfaceRef ioStats2 = make_surface(4096);
            id ioObj_out2 = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut2);
            id buf_out2 = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(g_Buffer,
                @selector(bufferWithIOSurfaceObject:symbolIndex:source:), ioObj_out2, @0, (long long)1);
            id outputSets2 = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(g_OutputSets,
                @selector(objectWithstatsSurRef:outputBuffer:), ioStats2, @[buf_out2]);

            // Step 0: Map IOSurfaces BEFORE preparing chaining
            {
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                id mapReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
                e = nil;
                BOOL mapOk = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                    mdl, @selector(mapIOSurfacesWithRequest:cacheInference:error:), mapReq, NO, &e);
                printf("  0. mapIOSurfaces (before prepare): %s", mapOk ? "OK" : "FAIL");
                if (!mapOk && e) printf(" err=%ld", (long)[e code]);
                printf("\n");
            }

            // Step 1: Prepare chaining — dual-procedure with loopback (output 0 → input 1)
            id cr = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(g_ChainingReq,
                @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                @[buf_in], @[outputSets, outputSets2], @[@0], @[@0], @0, @[], @1, @0, @0);

            e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                aneModel, @{}, cr, 21, &e);
            printf("  1. prepareChaining: %s\n", ok ? "OK" : "FAIL");
            if (!ok && e) printf("     Error: %s\n", [[e description] UTF8String]);

            if (ok) {
                // Step 2: Signal buffers ready
                id inputReady = ((id(*)(Class,SEL,unsigned int,id,id,uint64_t))objc_msgSend)(
                    g_InputReady,
                    @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                    0, @[@0], @[@0], (uint64_t)0);
                printf("  2. inputBuffersReady created: %s\n", inputReady ? "OK" : "nil");

                // Try multiple variations
                // Map IOSurfaces on the model (discovered: mapIOSurfacesWithRequest:cacheInference:error: on _ANEInMemoryModel)
                // Build a normal _ANERequest for the mapping
                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                id mapReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

                @try {
                    e = nil;
                    BOOL mapOk = ((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(
                        mdl, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                        mapReq, YES, &e);
                    printf("     mapIOSurfaces(model): %s", mapOk ? "OK" : "FAIL");
                    if (!mapOk && e) printf(" err=%ld", (long)[e code]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("     mapIOSurfaces: EXCEPTION %s\n", [[ex reason] UTF8String]);
                }

                // Variation A: doBuffersReady with _ANEModel — use NSError** properly
                NSError *berr = nil;
                @try {
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
                        aneModel, inputReady, @{}, 21, &berr);
                    printf("     A. doBuffersReady(_ANEModel): %s", ok ? "OK" : "FAIL");
                    if (!ok && berr) printf(" err=%ld: %s", (long)[berr code], [[berr localizedDescription] UTF8String]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("     A. doBuffersReady: EXCEPTION %s\n", [[ex reason] UTF8String]);
                }

                // Variation B: doBuffersReady with _ANEInMemoryModel
                e = nil;
                @try {
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(doBuffersReadyWithModel:inputBuffers:options:qos:error:),
                        mdl, inputReady, @{}, 21, &e);
                    printf("     B. doBuffersReady(_ANEInMemoryModel): %s", ok ? "OK" : "FAIL");
                    if (!ok && e) printf(" err=%ld", (long)[e code]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("     B. doBuffersReady(InMem): EXCEPTION %s\n", [[ex reason] UTF8String]);
                }

                // Variation C: daemon path with _ANEModel
                e = nil;
                @try {
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                        aneModel, inputReady, @{}, 21, &e);
                    printf("     C. buffersReady(daemon,_ANEModel): %s", ok ? "OK" : "FAIL");
                    if (!ok && e) printf(" err=%ld", (long)[e code]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("     C. buffersReady: EXCEPTION %s\n", [[ex reason] UTF8String]);
                }

                // Variation D: Try re-preparing with daemon path first, then buffersReady
                e = nil;
                @try {
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        aneModel, @{}, cr, 21, &e);
                    e = nil;
                    ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                        aneModel, inputReady, @{}, 21, &e);
                    printf("     D. daemon prepare+buffersReady: %s", ok ? "OK" : "FAIL");
                    if (!ok && e) printf(" err=%ld", (long)[e code]);
                    printf("\n");
                } @catch (NSException *ex) {
                    printf("     D. EXCEPTION %s\n", [[ex reason] UTF8String]);
                }

                if (ok) {
                    // Step 3: Enqueue output sets
                    id outputEnq = ((id(*)(Class,SEL,unsigned int,unsigned int,uint64_t,BOOL,BOOL))objc_msgSend)(
                        g_OutputEnq,
                        @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                        0, 0, (uint64_t)1, NO, NO);
                    printf("  3. outputSetEnqueue created: %s\n", outputEnq ? "OK" : "nil");

                    @try {
                        e = nil;
                        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(doEnqueueSetsWithModel:outputSet:options:qos:error:),
                            aneModel, outputEnq, @{}, 21, &e);
                        printf("     doEnqueueSets (direct): %s\n", ok ? "OK" : "FAIL");
                        if (!ok && e) printf("     Error: %s\n", [[e description] UTF8String]);
                    } @catch (NSException *ex) {
                        printf("     doEnqueueSets: EXCEPTION %s\n", [[ex reason] UTF8String]);
                        @try {
                            e = nil;
                            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                aneModel, outputEnq, @{}, 21, &e);
                            printf("     enqueueSets (daemon): %s\n", ok ? "OK" : "FAIL");
                            if (!ok && e) printf("     Error: %s\n", [[e description] UTF8String]);
                        } @catch (NSException *ex2) {
                            printf("     enqueueSets: EXCEPTION %s\n", [[ex2 reason] UTF8String]);
                        }
                    }

                    // Read output
                    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
                    printf("\n  Output[0..3] = [%.4f, %.4f, %.4f, %.4f]\n", out[0], out[1], out[2], out[3]);
                    bool isIdentity = fabsf(out[0] - 1.0f) < 0.05f;
                    printf("  Output %s (expected ~1.0 in channel 0)\n",
                           isIdentity ? "CORRECT! Chaining computed correctly!" : "WRONG — data not computed");
                    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

                    // ── Benchmark: chaining vs normal eval ──
                    if (isIdentity) {
                        printf("\n━━━ Benchmark: Chaining vs Normal ━━━\n");
                        int N = 100;

                        // Benchmark normal eval
                        id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                        id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                            @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

                        // Warmup
                        for (int i = 0; i < 5; i++)
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                                mdl, @{}, req, 21, &e);

                        uint64_t t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++)
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                                mdl, @{}, req, 21, &e);
                        uint64_t t1 = mach_absolute_time();
                        double normal_us = tb_us(t1 - t0) / N;
                        printf("  Normal eval:   %.1f μs/dispatch\n", normal_us);

                        // Benchmark chaining (prepare once, then buffersReady+enqueueSets N times)
                        // Re-prepare
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(doPrepareChainingWithModel:options:chainingReq:qos:error:),
                            aneModel, @{}, cr, 21, &e);

                        // Warmup chaining dispatch
                        for (int i = 0; i < 5; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                aneModel, inputReady, @{}, 21, &e);
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                aneModel, outputEnq, @{}, 21, &e);
                        }

                        t0 = mach_absolute_time();
                        for (int i = 0; i < N; i++) {
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                                aneModel, inputReady, @{}, 21, &e);
                            ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                                client, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                                aneModel, outputEnq, @{}, 21, &e);
                        }
                        t1 = mach_absolute_time();
                        double chain_us = tb_us(t1 - t0) / N;
                        printf("  Chained eval:  %.1f μs/dispatch\n", chain_us);
                        printf("  Speedup: %.2fx\n", normal_us / chain_us);
                    }
                }
            }

            CFRelease(ioIn); CFRelease(ioOut); CFRelease(ioStats);
        }

        [fm removeItemAtPath:td error:nil];
        [mdl release];
        [client release];

        printf("\nDone.\n");
    }
    return 0;
}
