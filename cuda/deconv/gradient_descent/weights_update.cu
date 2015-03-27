#ifndef DECONV_MODE
#error "DECONV_MODE must be defined"
#endif
#if ((DECONV_MODE != 1) && (DECONV_MODE != 2))
#error "DECONV_MODE must be equal to 1 or 2"
#endif

extern "C"
__global__ void err_output_update(dtype *err_output
#if DECONV_MODE == 2
                                  , const int *hits
#endif
                                  ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
#if DECONV_MODE == 2
    int n = hits[idx];
    err_output[idx] /= n ? n : 1;
#else
    err_output[idx] /= (KX / SLIDE_X) * (KY / SLIDE_Y);
#endif
  }
}

#include "all2all/gradient_descent/weights_update.cu"
