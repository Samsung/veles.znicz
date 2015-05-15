#ifndef DECONV_MODE
#error "DECONV_MODE must be defined"
#endif
#if ((DECONV_MODE != 1) && (DECONV_MODE != 2))
#error "DECONV_MODE must be equal to 1 or 2"
#endif

#include "conv/gradient_descent/err_input_update.cl"

#if DECONV_MODE == 2
__kernel void apply_hits(__global dtype *output, __global const int *hits) {
  int idx = get_global_id(0);
  int n = hits[idx];
  output[idx] /= n ? n : 1;
}

KERNEL_CLEAR(clear_hits, int)
#endif

KERNEL_CLEAR(clear_output, dtype)
