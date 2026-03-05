# Apple Neural Engine Internals — What maderix Didn't Cover

> Findings from 7 probe iterations + CoreML MLProgram reverse engineering on Apple M5, macOS 26.3.
> Everything here is novel — not documented in [maderix/ANE](https://github.com/maderix/ANE), [hollance/neural-engine](https://github.com/hollance/neural-engine), or any public source.

---

## 1. _ANEInMemoryModel Internal Structure

maderix showed how to create and compile `_ANEInMemoryModel`. We mapped its full ivar layout:

```
Offset  Type                          Field
[+8]    char                          _queueDepth = 127
[+9]    BOOL                          _isMILModel
[+12]   uint32                        _perfStatsMask
[+16]   NSDictionary                  _modelAttributes
[+24]   NSString                      _hexStringIdentifier
[+32]   uint64                        _intermediateBufferHandle
[+40]   _ANEClient                    _sharedConnection
[+48]   NSURL                         _modelURL
[+56]   NSString                      _compilerOptionsFileName
[+64]   _ANEModel                     _model (the kernel handle)
[+72]   uint64                        _string_id
[+80]   uint64                        _programHandle
[+88]   _ANEProgramForEvaluation      _program
[+96]   uint64                        _state (3 = loaded)
[+104]  _ANEInMemoryModelDescriptor   _descriptor
```

Key insight: `_intermediateBufferHandle` at +32 is likely required for chaining — it's the handle for intermediate buffers between chained procedures.

---

## 2. Multi-Procedure Models

maderix only used single-function MIL programs (`func main<ios18>`). We discovered you can put **multiple functions in one MIL program** and address them by `procedureIndex`:

```
program(1.3) [...] {
    func layer0<ios18>(tensor<fp32, [1, 64, 1, 32]> x0) { ... } -> (y0);
    func layer1<ios18>(tensor<fp32, [1, 64, 1, 32]> x1) { ... } -> (y1);
}
```

The compiled model reports:
```
ANEFModelProcedures: [
  { ProcedureID: 0, InputSymbolIndexArray: [0], OutputSymbolIndexArray: [0] },
  { ProcedureID: 1, InputSymbolIndexArray: [1], OutputSymbolIndexArray: [1] }
]
ProcedureNameToIDMap: { layer0: 0, layer1: 1 }
InputSymbols: [x0, x1]
OutputSymbols: [y0@output, y1@output]
```

**Critical:** Procedure N uses `inputIndex=N, outputIndex=N` (not 0). Using wrong indices → error 0x2.

Read from `modelAttributes["ANEFModelDescription"]["ANEFModelProcedures"]`.

---

## 3. Direct Eval — Bypassing the ANE Daemon

maderix uses `evaluateWithQoS:options:request:error:` which goes through the ANE daemon via XPC. We found a **10% faster path**:

```objc
// Acquire direct client
_ANEClient *client = [[_ANEClient alloc] initWithRestrictedAccessAllowed:YES];

// Direct eval — bypasses daemon XPC
[client doEvaluateDirectWithModel:model
                          options:@{}
                          request:request
                              qos:21
                            error:&error];
```

| Path | Latency |
|------|---------|
| `evaluateWithQoS:` (standard) | 117 μs |
| `doEvaluateDirectWithModel:` (direct) | **106 μs** |

The `do` prefix methods bypass the daemon entirely — the client talks directly to the kernel driver.

---

## 4. Complete _ANEClient API Surface

maderix documented `evaluateWithQoS:` and `compileWithQoS:`. Here's the full method list:

```objc
// Evaluation
-evaluateWithModel:options:request:qos:error:        // Standard (daemon)
-doEvaluateDirectWithModel:options:request:qos:error: // Direct (10% faster)
-evaluateRealTimeWithModel:options:request:error:     // RT path (entitlement gated)

// Model lifecycle
-compileModel:options:qos:error:
-loadModel:options:qos:error:
-unloadModel:options:qos:error:
-doLoadModel:options:qos:error:                       // Direct load
-doUnloadModel:options:qos:error:                     // Direct unload
-loadModelNewInstance:options:modelInstParams:qos:error:
-doLoadModelNewInstance:options:modelInstParams:qos:error:
-loadRealTimeModel:options:qos:error:
-unloadRealTimeModel:options:qos:error:

// Chaining
-prepareChainingWithModel:options:chainingReq:qos:error:
-doPrepareChainingWithModel:options:chainingReq:qos:error:
-buffersReadyWithModel:inputBuffers:options:qos:error:
-doBuffersReadyWithModel:inputBuffers:options:qos:error:
-enqueueSetsWithModel:outputSet:options:qos:error:
-doEnqueueSetsWithModel:outputSet:options:qos:error:

// Session
-sessionHintWithModel:hint:options:report:error:      // hint is NSString
-reportEvaluateFailure:failureReason:qIdx:

// Introspection
-compiledModelExistsFor:
-compiledModelExistsMatchingHash:
-purgeCompiledModel:
-purgeCompiledModelMatchingHash:
-connections
-connectionsUsedForLoadingModels
-connectionForLoadingModel:options:
-connectionUsedForLoadingModel:
-isAnetoolRootDaemonConnection
```

---

## 5. Chaining API — Full Surface

maderix never explored chaining. We mapped 10 related classes:

### _ANEChainingRequest
```objc
+chainingRequestWithInputs:(NSArray<_ANEBuffer>)
                outputSets:(NSArray<_ANEIOSurfaceOutputSets>)
           lbInputSymbolId:(NSArray<NSNumber>)
          lbOutputSymbolId:(NSArray<NSNumber>)
            procedureIndex:(NSNumber)
              signalEvents:(NSArray)
         transactionHandle:(NSNumber)
           fwEnqueueDelay:(NSNumber)
              memoryPoolId:(NSNumber)

-validate → BOOL (returns YES for valid requests)
```

### _ANEBuffer
```objc
+bufferWithIOSurfaceObject:(ANEIOSurfaceObject)
               symbolIndex:(NSNumber)
                    source:(long long)   // 0=input, 1=output
```

### _ANEIOSurfaceOutputSets
```objc
// THE CORRECT FACTORY METHOD (took 7 probes to find):
+objectWithstatsSurRef:(IOSurfaceRef)outputBuffer:(NSArray<_ANEBuffer>)

// NOT this (doesn't exist, caused error 15):
// +outputSetsWithBuffers:  ← WRONG
```

### _ANEInputBuffersReady
```objc
+inputBuffersWithProcedureIndex:(uint)
            inputBufferInfoIndex:(NSArray<NSNumber>)
                  inputFreeValue:(NSArray<NSNumber>)
                  executionDelay:(uint64)
```

### _ANEOutputSetEnqueue
```objc
+outputSetWithProcedureIndex:(uint)
                    setIndex:(uint)
                 signalValue:(uint64)
           signalNotRequired:(BOOL)
                  isOpenLoop:(BOOL)
```

### Chaining Sequence
```
1. prepareChainingWithModel:  → sets up firmware pipeline
2. buffersReadyWithModel:     → signals input data is ready
3. enqueueSetsWithModel:      → enqueues output collection
```

**Status:** `prepareChaining` succeeds (both daemon and direct paths). `buffersReady` fails silently. The sequence is mandatory — can't skip steps.

---

## 6. _ANEModel vs _ANEInMemoryModel

maderix works with `_ANEInMemoryModel` but never explains the relationship to `_ANEModel`:

- `_ANEInMemoryModel` is the ObjC wrapper (memory management, compilation, temp dirs)
- `_ANEModel` is the kernel handle (UUID, programHandle, modelAttributes)
- Access via: `[inMemModel model]` → returns `_ANEModel`
- Chaining APIs need `_ANEModel`, not `_ANEInMemoryModel`
- The model's own `_sharedConnection` is an `_ANEClient` but chaining with it fails (error 15) — need a fresh `_ANEClient`

---

## 7. IOKit — H11ANE Kernel Driver

Not covered by any public source. We probed the IOKit registry:

```
IOService tree (ANE entries):
  ane                    (service)
  dart-ane               (DMA address remapping)
  mapper-ane             (memory mapper)
  mapper-ane-mpm         (memory pool manager)
  ANEDriverRoot          (H1xANELoadBalancer)
  H11ANE                 (H11ANEIn — main driver)
  AppleT8132ANEHAL       (hardware abstraction layer)
  ANEClientHints         (client hint service)
```

### User Client Types
```
IOServiceOpen(H11ANE, type=0) → FAIL (unsupported)
IOServiceOpen(H11ANE, type=1) → SUCCESS (H11ANEInDirectPathClient)
IOServiceOpen(H11ANE, type=4) → SUCCESS (unknown client type)
IOServiceOpen(ANEDriverRoot, type=1) → SUCCESS (load balancer client)
```

From [Phrack](https://phrack.org/issues/72/9_md) and [Project Zero](https://projectzero.google/2020/11/oops-i-missed-it-again.html):
- **50 external method selectors** on H11ANEInUserClient
- **H11ANEInDirectPathClient** (type=1): less privileged, originally limited to 3 methods
- **H11ANEInUserClient** (type=0): full entitlement, 34 methods
- Selector 2: `ANE_ProgramSendRequest_gated()` (inputStructSize=2376)
- Historical bug: DirectPathClient could access UserClient methods due to copy-paste bounds-check error (CVE-2020-27905)

---

## 8. CoreML MLProgram → Espresso Runtime

Not documented anywhere. We compiled an MLProgram `.mlpackage` and deep-walked the loaded model:

```
MLModel
  └─ _internalEngine: MLMultiFunctionProgramEngine
       └─ _functionNameToEngineMap: { "main" → MLProgramEngine }
            └─ MLProgramEngine (subclass of MLNeuralNetworkEngine)
```

### MLNeuralNetworkEngine — 52 ivars mapped

Key fields:
```
_isANEPathForbidden (BOOL) = NO     ← ANE IS enabled for MLProgram
_isGPUPathForbidden (BOOL) = NO
_modelIsMIL (BOOL) = YES
_usingCPU (BOOL) = NO               ← NOT running on CPU
_plan (void*) = Espresso plan ptr    ← C++ execution plan
_context (void*) = Espresso context  ← C++ runtime context
_compilerOutput = nil                ← lazy compilation
_inputLayers = [...]
_outputLayers = [...]
_network = { plan: ptr, network_index: 0 }
```

### Conclusion
CoreML MLProgram uses **Espresso** (Apple's internal C++ ML runtime) for ANE dispatch. The ANE model is inside `_plan` and `_context` (opaque C++ pointers), NOT exposed as an `_ANEModel` ObjC object. This means CoreML models and private-API models use **completely separate compilation and dispatch pipelines** to the same ANE hardware.

---

## 9. MLModelConfiguration — 32 Properties

Undocumented properties on `MLModelConfiguration`:

```
experimentalMLE5EngineUsage        ← E5 runtime flag
e5rtComputeDeviceTypeMask          ← device targeting
e5rtCustomANECompilerOptions       ← string, nil by default
e5rtDynamicCallableFunctions       ← dynamic dispatch
neuralEngineCompilerOptions        ← "EnableLowEffortCPAllocation=true"
computeUnits                       ← MLComputeUnitsAll
allowLowPrecisionAccumulationOnGPU
optimizationHints
modelDisplayName
... (32 total)
```

Setting `experimentalMLE5EngineUsage = 1` routes through the E5/Espresso pipeline. `neuralEngineCompilerOptions` is a string format (not dict) for ANE compiler flags.

---

## 10. _ANEDeviceController

```objc
-device → ANEDeviceStruct* { void*, void*, void*, char, int, uint64 }
-setDevice:(ANEDeviceStruct*)
-start / -stop
-programHandle → uint64
-initWithProgramHandle:priviledged:
-initWithANEPrivilegedVM:(BOOL)
-isPrivileged → BOOL
-usecount / -setUsecount:
```

The `ANEDeviceStruct` contains raw kernel handles (3 void pointers). This is the lowest-level ObjC interface before hitting IOKit directly.

---

## Summary: What's New vs maderix

| Topic | maderix/ANE | ane-infer (this project) |
|-------|-------------|--------------------------|
| `_ANEInMemoryModel` usage | Basic compile/eval | Full 13-ivar layout mapped with offsets |
| Multi-procedure MIL | Not explored | Working — N functions, procedureIndex dispatch |
| Direct eval path | Not explored | `doEvaluateDirectWithModel:` — 10% faster |
| Chaining API | Listed as "unexplored" | 10 classes fully mapped, `prepareChaining` succeeds |
| `_ANEIOSurfaceOutputSets` | Not documented | Correct factory: `objectWithstatsSurRef:outputBuffer:` |
| CoreML MLProgram internals | Not explored | Full engine hierarchy: MLProgramEngine → MLNeuralNetworkEngine |
| Espresso runtime | Not mentioned | Identified as CoreML's parallel ANE dispatch pipeline |
| IOKit H11ANE driver | Not explored | 3 user client types, 50 selectors, Phrack/P0 cross-reference |
| _ANEClient full API | 3 methods shown | 25 methods documented |
| MLModelConfiguration | Not explored | 32 properties, E5 runtime flags, compiler options |
| _ANEDeviceController | Not explored | Full API + ANEDeviceStruct layout |
| Session hints | Not explored | NSString type confirmed, 8 hints tested |
| Real-time path | Not explored | Entitlement-gated (`beginRealTimeTask` → NO) |
