// test_fused_ffn.m — Test fused FFN mega-kernel compilation and execution on ANE
// Build: xcrun clang -O2 -fno-objc-arc -o /tmp/test_fused_ffn test_fused_ffn.m \
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

// Build a multi-chunk weight blob: [global_hdr(64)] [chunk0_hdr(64) + data] [chunk1_hdr(64) + data] ...
static NSData *build_weight_blob(int n_chunks, int *chunk_fp16_sizes, _Float16 **chunk_data) {
    // Calculate total size
    int total = 64; // global header
    for (int i = 0; i < n_chunks; i++) total += 64 + chunk_fp16_sizes[i];

    uint8_t *blob = (uint8_t*)calloc(total, 1);
    blob[0] = 1; blob[4] = 2; // global header magic

    int offset = 64;
    for (int i = 0; i < n_chunks; i++) {
        int data_sz = chunk_fp16_sizes[i];
        // Chunk header
        blob[offset] = 0xEF; blob[offset+1] = 0xBE; blob[offset+2] = 0xAD; blob[offset+3] = 0xDE;
        blob[offset+4] = 1; // version
        *(uint32_t*)(blob + offset + 8) = data_sz; // data size
        *(uint32_t*)(blob + offset + 16) = offset + 64; // absolute data offset

        // Copy FP16 data
        memcpy(blob + offset + 64, chunk_data[i], data_sz);
        offset += 64 + data_sz;
    }

    return [NSData dataWithBytesNoCopy:blob length:total freeWhenDone:YES];
}

int main() {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        Class g_D = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class g_I = NSClassFromString(@"_ANEInMemoryModel");
        Class g_AR = NSClassFromString(@"_ANERequest");
        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class g_Client = NSClassFromString(@"_ANEClient");

        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║  Fused FFN Mega-Kernel Test                         ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n\n");

        // Test with small dimensions first, then scale up
        int test_configs[][3] = {
            {64, 128, 16},    // tiny: fits easily
            {256, 512, 32},   // small
            {512, 1024, 32},  // medium
            {2048, 6144, 16}, // Qwen3.5-2B actual dims, small spatial
            {2048, 6144, 64}, // Qwen3.5-2B actual dims, larger spatial
        };

        id client = ((id(*)(id,SEL,BOOL))objc_msgSend)(
            [g_Client alloc], @selector(initWithRestrictedAccessAllowed:), YES);

        for (int tc = 0; tc < 5; tc++) {
            int dim = test_configs[tc][0];
            int hidden = test_configs[tc][1];
            int spatial = test_configs[tc][2];

            printf("━━━ Test %d: dim=%d hidden=%d spatial=%d ━━━\n", tc, dim, hidden, spatial);

            // Weight sizes in FP16 bytes
            int gate_sz = hidden * dim * 2;  // [hidden, dim] FP16
            int up_sz = hidden * dim * 2;
            int down_sz = dim * hidden * 2;
            int total_weight_mb = (gate_sz + up_sz + down_sz) / (1024*1024);
            printf("  Weights: gate=%dMB up=%dMB down=%dMB total=%dMB\n",
                   gate_sz/(1024*1024), up_sz/(1024*1024), down_sz/(1024*1024), total_weight_mb);

            // Build identity-ish weights (small random values + diagonal)
            _Float16 *w_gate = (_Float16*)calloc(hidden * dim, sizeof(_Float16));
            _Float16 *w_up = (_Float16*)calloc(hidden * dim, sizeof(_Float16));
            _Float16 *w_down = (_Float16*)calloc(dim * hidden, sizeof(_Float16));
            int min_dim = dim < hidden ? dim : hidden;
            for (int i = 0; i < min_dim; i++) {
                w_gate[i * dim + i] = (_Float16)0.1f;
                w_up[i * dim + i] = (_Float16)1.0f;
                w_down[i * hidden + i] = (_Float16)1.0f;
            }

            _Float16 *chunks[] = {w_gate, w_up, w_down};
            int sizes[] = {gate_sz, up_sz, down_sz};
            NSData *wdata = build_weight_blob(3, sizes, chunks);

            // Build fused FFN MIL
            int cs_gate = 64 + gate_sz;
            int cs_up = 64 + up_sz;
            uint64_t gate_off = 64;
            uint64_t up_off = 64 + cs_gate;
            uint64_t down_off = 64 + cs_gate + cs_up;

            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
                "        string c_pad_type = const()[name=string(\"c_pad_type\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> c_strides = const()[name=string(\"c_strides\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> c_pad = const()[name=string(\"c_pad\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> c_dilations = const()[name=string(\"c_dilations\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 c_groups = const()[name=string(\"c_groups\"), val=int32(1)];\n"
                "        string to_fp16 = const()[name=string(\"to_fp16\"), val=string(\"fp16\")];\n"
                "        string to_fp32 = const()[name=string(\"to_fp32\"), val=string(\"fp32\")];\n"
                "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to_fp16, x=x)[name=string(\"cast_in\")];\n"
                // gate_proj conv
                "        tensor<fp16, [%d,%d,1,1]> W_gate = const()[name=string(\"W_gate\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=c_dilations, groups=c_groups, pad=c_pad, pad_type=c_pad_type, strides=c_strides, weight=W_gate, x=x16)[name=string(\"conv_gate\")];\n"
                // SiLU
                "        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sigmoid\")];\n"
                "        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1, y=sig)[name=string(\"silu\")];\n"
                // up_proj conv
                "        tensor<fp16, [%d,%d,1,1]> W_up = const()[name=string(\"W_up\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=c_dilations, groups=c_groups, pad=c_pad, pad_type=c_pad_type, strides=c_strides, weight=W_up, x=x16)[name=string(\"conv_up\")];\n"
                // gated = silu * h3
                "        tensor<fp16, [1,%d,1,%d]> gated = mul(x=silu, y=h3)[name=string(\"gate_mul\")];\n"
                // down_proj conv
                "        tensor<fp16, [%d,%d,1,1]> W_down = const()[name=string(\"W_down\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(%llu)))];\n"
                "        tensor<fp16, [1,%d,1,%d]> out16 = conv(dilations=c_dilations, groups=c_groups, pad=c_pad, pad_type=c_pad_type, strides=c_strides, weight=W_down, x=gated)[name=string(\"conv_down\")];\n"
                // Cast output
                "        tensor<fp32, [1,%d,1,%d]> y = cast(dtype=to_fp32, x=out16)[name=string(\"cast_out\")];\n"
                "    } -> (y);\n"
                "}\n",
                dim, spatial,
                dim, spatial,
                hidden, dim, hidden, dim, (unsigned long long)gate_off,
                hidden, spatial,
                hidden, spatial,
                hidden, spatial,
                hidden, dim, hidden, dim, (unsigned long long)up_off,
                hidden, spatial,
                hidden, spatial,
                dim, hidden, dim, hidden, (unsigned long long)down_off,
                dim, spatial,
                dim, spatial];

            NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
            NSDictionary *wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wdata}};

            // Compile
            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D,
                @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
            if (!desc) { printf("  FAIL: descriptor creation\n"); goto next; }

            id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I,
                @selector(inMemoryModelWithDescriptor:), desc);
            if (!mdl) { printf("  FAIL: model creation\n"); goto next; }

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
            printf("  Compile: %s", ok ? "OK" : "FAIL");
            if (!ok && e) printf(" (%s)", [[e localizedDescription] UTF8String]);
            printf("\n");
            if (!ok) { [fm removeItemAtPath:td error:nil]; goto next; }

            ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            printf("  Load: %s", ok ? "OK" : "FAIL");
            if (!ok && e) printf(" (%s)", [[e localizedDescription] UTF8String]);
            printf("\n");
            if (!ok) { [fm removeItemAtPath:td error:nil]; goto next; }

            // Execute
            {
                int in_bytes = dim * spatial * 4;
                int out_bytes = dim * spatial * 4;
                IOSurfaceRef ioIn = make_surface(in_bytes);
                IOSurfaceRef ioOut = make_surface(out_bytes);

                // Fill input: channel-first [1, dim, 1, spatial], set channel 0 = 1.0
                IOSurfaceLock(ioIn, 0, NULL);
                float *inp = (float*)IOSurfaceGetBaseAddress(ioIn);
                memset(inp, 0, in_bytes);
                for (int s = 0; s < spatial; s++) inp[0*spatial+s] = 1.0f; // channel 0
                IOSurfaceUnlock(ioIn, 0, NULL);

                id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
                id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

                e = nil;
                ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                    mdl, @{}, req, 21, &e);
                printf("  Execute: %s", ok ? "OK" : "FAIL");
                if (!ok && e) printf(" (%s)", [[e localizedDescription] UTF8String]);
                printf("\n");

                if (ok) {
                    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
                    float *out = (float*)IOSurfaceGetBaseAddress(ioOut);
                    printf("  Output[0..3] = [%.6f, %.6f, %.6f, %.6f]\n", out[0], out[1], out[2], out[3]);
                    float max_val = 0;
                    for (int i = 0; i < dim*spatial; i++) if (fabsf(out[i]) > max_val) max_val = fabsf(out[i]);
                    printf("  Max |output| = %.6f\n", max_val);
                    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

                    // Benchmark
                    for (int i = 0; i < 3; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            mdl, @{}, req, 21, &e);
                    int N = 20;
                    uint64_t t0 = mach_absolute_time();
                    for (int i = 0; i < N; i++)
                        ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                            client, @selector(doEvaluateDirectWithModel:options:request:qos:error:),
                            mdl, @{}, req, 21, &e);
                    uint64_t t1 = mach_absolute_time();
                    double us = tb_us(t1 - t0) / N;
                    // FLOPs: 3 convs + 2 element-wise ops
                    double flops = (2.0*hidden*dim + 2.0*hidden*dim + 2.0*dim*hidden) * spatial; // 3 matmuls
                    double tflops = flops / (us * 1e-6) / 1e12;
                    printf("  Latency: %.1f μs  (%.2f TFLOPS)\n", us, tflops);
                }

                CFRelease(ioIn); CFRelease(ioOut);
            }

            [fm removeItemAtPath:td error:nil];
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e);
            next:
            free(w_gate); free(w_up); free(w_down);
            printf("\n");
        }

        [client release];
        printf("Done.\n");
    }
    return 0;
}
