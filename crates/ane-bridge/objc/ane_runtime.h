// ane_runtime.h — C ABI for ANE in-memory compile/eval/free
// Adapted from maderix/ANE for Rust FFI
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a compiled ANE kernel
typedef struct ANEKernel ANEKernel;

// Initialize ANE runtime (loads private framework). Call once.
// Returns 0 on success, -1 on failure.
int ane_init(void);

// Compile MIL text + weight blob into an ANE kernel.
// mil_text: UTF-8 MIL program text
// mil_len: length of mil_text in bytes
// weight_data: raw weight blob (NULL if no weights)
// weight_len: length of weight_data in bytes
// n_inputs: number of input tensors
// input_sizes: array of byte sizes per input tensor
// n_outputs: number of output tensors
// output_sizes: array of byte sizes per output tensor
// Returns NULL on failure.
ANEKernel *ane_compile(const char *mil_text, size_t mil_len,
                       const uint8_t *weight_data, size_t weight_len,
                       int n_inputs, const size_t *input_sizes,
                       int n_outputs, const size_t *output_sizes);

// Write data to an input IOSurface.
void ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes);

// Read data from an output IOSurface.
void ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes);

// Execute the compiled kernel on ANE.
// Returns true on success.
bool ane_eval(ANEKernel *k);

// Execute a specific procedure within a multi-procedure model.
// proc_idx: 0-based procedure index
// Returns true on success.
bool ane_eval_procedure(ANEKernel *k, int proc_idx);

// Get the number of procedures in a compiled model.
int ane_num_procedures(ANEKernel *k);

// Resize input/output IOSurfaces without recompiling.
// Useful for multi-procedure models where different procedures need different sizes.
void ane_resize_io(ANEKernel *k, int n_inputs, const size_t *input_sizes,
                   int n_outputs, const size_t *output_sizes);

// Free all resources associated with a kernel.
void ane_free(ANEKernel *k);

#ifdef __cplusplus
}
#endif
