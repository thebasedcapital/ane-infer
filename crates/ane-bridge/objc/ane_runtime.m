// ane_runtime.m — C ABI implementation for ANE in-memory model lifecycle
// Adapted from maderix/ANE training/ane_runtime.h
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include "ane_runtime.h"

// ANE private framework classes (resolved at runtime)
static Class g_ANEDesc   = nil;  // _ANEInMemoryModelDescriptor
static Class g_ANEInMem  = nil;  // _ANEInMemoryModel
static Class g_ANEReq    = nil;  // _ANERequest
static Class g_ANEIO     = nil;  // _ANEIOSurfaceObject
static Class g_ANEClient = nil;  // _ANEClient (for direct eval bypass)
static id    g_client    = nil;  // shared _ANEClient instance
static bool  g_loaded    = false;

struct ANEKernel {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest (for procedure 0)
    id *procRequests;       // pre-built requests per procedure (NULL if single-proc)
    int nProcs;             // number of procedures
    NSString *tmpDir;
    int nInputs;
    int nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
};

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:           @(bytes),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:     @(bytes),
        (id)kIOSurfaceAllocSize:       @(bytes),
        (id)kIOSurfacePixelFormat:     @0
    });
}

int ane_init(void) {
    if (g_loaded) return 0;

    void *handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ane_init: failed to load AppleNeuralEngine.framework: %s\n", dlerror());
        return -1;
    }

    g_ANEDesc   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem  = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq    = NSClassFromString(@"_ANERequest");
    g_ANEIO     = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ANEClient = NSClassFromString(@"_ANEClient");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_init: failed to resolve private classes\n");
        return -1;
    }

    // Acquire direct ANE client (bypasses daemon for ~10% faster eval)
    if (g_ANEClient) {
        g_client = ((id(*)(id,SEL,BOOL))objc_msgSend)(
            [g_ANEClient alloc],
            @selector(initWithRestrictedAccessAllowed:),
            YES);
        if (g_client) {
            [g_client retain];
            fprintf(stderr, "ane_init: direct eval path active (_ANEClient acquired)\n");
        } else {
            fprintf(stderr, "ane_init: warning: _ANEClient init failed, using daemon path\n");
        }
    }

    g_loaded = true;
    return 0;
}

ANEKernel *ane_compile(const char *mil_text, size_t mil_len,
                       const uint8_t *weight_data, size_t weight_len,
                       int n_inputs, const size_t *input_sizes,
                       int n_outputs, const size_t *output_sizes) {
    if (!g_loaded && ane_init() != 0) return NULL;

    @autoreleasepool {
        NSError *error = nil;
        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];

        // Build weight dictionary if weights provided
        NSDictionary *wdict = nil;
        if (weight_data && weight_len > 0) {
            NSData *wData = [NSData dataWithBytes:weight_data length:weight_len];
            wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wData}};
        }

        // Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc,
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "ane_compile: modelWithMILText failed\n");
            return NULL;
        }

        // Create in-memory model
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem,
            @selector(inMemoryModelWithDescriptor:),
            desc);
        if (!mdl) {
            fprintf(stderr, "ane_compile: inMemoryModelWithDescriptor failed\n");
            return NULL;
        }

        // Write MIL + weights to temp dir (required by compiler)
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weight_data && weight_len > 0) {
            NSData *wData = [NSData dataWithBytes:weight_data length:weight_len];
            [wData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"]
                    atomically:YES];
        }

        // Compile (QoS 21)
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &error)) {
            fprintf(stderr, "ane_compile: compile failed: %s\n",
                    [[error description] UTF8String]);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Load onto ANE
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &error)) {
            fprintf(stderr, "ane_compile: load failed: %s\n",
                    [[error description] UTF8String]);
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }

        // Allocate kernel struct
        ANEKernel *k = (ANEKernel *)calloc(1, sizeof(ANEKernel));
        k->model = [mdl retain];
        k->tmpDir = [td retain];
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        // Create IOSurfaces for I/O
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs; i++)
            k->ioInputs[i] = create_surface(input_sizes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_sizes[i]);

        // Build ANE request (reused across evals)
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);
        k->request = [req retain];

        // Pre-build requests for multi-procedure models
        k->nProcs = 0;
        k->procRequests = NULL;

        // Check how many procedures this model has
        id aneModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        if (aneModel && [aneModel respondsToSelector:@selector(modelAttributes)]) {
            id attrs = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(modelAttributes));
            if (attrs) {
                NSDictionary *fDesc = [attrs objectForKey:@"ANEFModelDescription"];
                if (fDesc) {
                    NSArray *procs = [fDesc objectForKey:@"ANEFModelProcedures"];
                    if (procs && [procs count] > 1) {
                        int np = (int)[procs count];
                        k->nProcs = np;
                        k->procRequests = (id *)calloc(np, sizeof(id));
                        for (int p = 0; p < np; p++) {
                            NSMutableArray *pIns = [NSMutableArray arrayWithCapacity:n_inputs];
                            NSMutableArray *pInIdx = [NSMutableArray arrayWithCapacity:n_inputs];
                            for (int j = 0; j < n_inputs; j++) {
                                [pIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                                    g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[j])];
                                [pInIdx addObject:@(p)];
                            }
                            NSMutableArray *pOuts = [NSMutableArray arrayWithCapacity:n_outputs];
                            NSMutableArray *pOutIdx = [NSMutableArray arrayWithCapacity:n_outputs];
                            for (int j = 0; j < n_outputs; j++) {
                                [pOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                                    g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[j])];
                                [pOutIdx addObject:@(p)];
                            }
                            id pReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                                g_ANEReq,
                                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                                pIns, pInIdx, pOuts, pOutIdx, nil, nil, @(p));
                            k->procRequests[p] = [pReq retain];
                        }
                    }
                }
            }
        }

        return k;
    }
}

void ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes) {
    if (!k || idx < 0 || idx >= k->nInputs) return;
    IOSurfaceLock(k->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(k->ioInputs[idx], 0, NULL);
}

void ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes) {
    if (!k || idx < 0 || idx >= k->nOutputs) return;
    IOSurfaceLock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

bool ane_eval(ANEKernel *k) {
    if (!k) return false;
    NSError *error = nil;
    BOOL ok;

    if (g_client) {
        // Direct eval path — bypasses ANE daemon for ~10% lower latency
        // doEvaluateDirectWithModel:options:request:qos:error:
        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client,
            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
            k->model, @{}, k->request, 21, &error);
    } else {
        // Fallback: daemon path
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, k->request, &error);
    }

    if (!ok) {
        fprintf(stderr, "ane_eval: evaluate failed: %s\n",
                [[error description] UTF8String]);
    }
    return ok;
}

int ane_num_procedures(ANEKernel *k) {
    if (!k) return 0;
    // Query modelAttributes → ANEFModelDescription → ANEFModelProcedures
    id aneModel = ((id(*)(id,SEL))objc_msgSend)(k->model, @selector(model));
    if (!aneModel) return 1;
    id attrs = ((id(*)(id,SEL))objc_msgSend)(aneModel, @selector(modelAttributes));
    if (!attrs) return 1;
    NSDictionary *fDesc = [attrs objectForKey:@"ANEFModelDescription"];
    if (!fDesc) return 1;
    NSArray *procs = [fDesc objectForKey:@"ANEFModelProcedures"];
    if (!procs) return 1;
    return (int)[procs count];
}

bool ane_eval_procedure(ANEKernel *k, int proc_idx) {
    if (!k) return false;
    NSError *error = nil;

    // Use pre-built request if available, otherwise fall back to building one
    id req;
    if (k->procRequests && proc_idx < k->nProcs) {
        req = k->procRequests[proc_idx];
    } else if (proc_idx == 0) {
        req = k->request;
    } else {
        // Fallback: build request on the fly (should not happen with pre-built)
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:k->nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:k->nInputs];
        for (int i = 0; i < k->nInputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(proc_idx)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:k->nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:k->nOutputs];
        for (int i = 0; i < k->nOutputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(proc_idx)];
        }
        req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @(proc_idx));
    }

    BOOL ok;
    if (g_client) {
        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            g_client,
            @selector(doEvaluateDirectWithModel:options:request:qos:error:),
            k->model, @{}, req, 21, &error);
    } else {
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model,
            @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &error);
    }

    if (!ok) {
        fprintf(stderr, "ane_eval_procedure(%d): failed: %s\n",
                proc_idx, [[error description] UTF8String]);
    }
    return ok;
}

void ane_resize_io(ANEKernel *k, int n_inputs, const size_t *input_sizes,
                   int n_outputs, const size_t *output_sizes) {
    if (!k) return;

    // Only recreate surfaces that changed size
    for (int i = 0; i < n_inputs && i < k->nInputs; i++) {
        if (input_sizes[i] != k->inputBytes[i]) {
            CFRelease(k->ioInputs[i]);
            k->ioInputs[i] = create_surface(input_sizes[i]);
            k->inputBytes[i] = input_sizes[i];
        }
    }
    for (int i = 0; i < n_outputs && i < k->nOutputs; i++) {
        if (output_sizes[i] != k->outputBytes[i]) {
            CFRelease(k->ioOutputs[i]);
            k->ioOutputs[i] = create_surface(output_sizes[i]);
            k->outputBytes[i] = output_sizes[i];
        }
    }

    // Rebuild the default request with new surfaces
    [k->request release];
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:k->nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:k->nInputs];
    for (int i = 0; i < k->nInputs; i++) {
        [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:k->nOutputs];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:k->nOutputs];
    for (int i = 0; i < k->nOutputs; i++) {
        [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
        [oIdx addObject:@(i)];
    }
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);
    k->request = [req retain];
}

void ane_free(ANEKernel *k) {
    if (!k) return;

    // Unload from ANE
    NSError *error = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &error);

    // Release IOSurfaces
    for (int i = 0; i < k->nInputs; i++)
        CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++)
        CFRelease(k->ioOutputs[i]);

    // Clean up temp directory
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];

    // Release ObjC objects
    [k->model release];
    [k->tmpDir release];
    [k->request release];
    if (k->procRequests) {
        for (int i = 0; i < k->nProcs; i++) {
            if (k->procRequests[i]) [k->procRequests[i] release];
        }
        free(k->procRequests);
    }

    free(k->ioInputs);
    free(k->ioOutputs);
    free(k->inputBytes);
    free(k->outputBytes);
    free(k);
}
