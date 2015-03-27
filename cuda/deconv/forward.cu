#ifndef DECONV_MODE
#error "DECONV_MODE must be defined"
#endif
#if ((DECONV_MODE != 1) && (DECONV_MODE != 2))
#error "DECONV_MODE must be equal to 1 or 2"
#endif

#include "conv/gradient_descent/err_input_update.cu"

#if DECONV_MODE == 2
extern "C"
__global__ void apply_hits(dtype *output, const int *hits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    int n = hits[idx];
    output[idx] /= n ? n : 1;
  }
}
#endif
