#ifndef DECONV_MODE
#error "DECONV_MODE must be defined"
#endif
#if ((DECONV_MODE != 1) && (DECONV_MODE != 2))
#error "DECONV_MODE must be equal to 1 or 2"
#endif

__kernel void err_output_update(__global dtype *err_output
#if DECONV_MODE == 2
                                , __global const int *hits
#endif
                                ) {
  int idx = get_global_id(0);
#if DECONV_MODE == 2
    int n = hits[idx];
    err_output[idx] /= n ? n : 1;
#else
    err_output[idx] /= (KX / SLIDE_X) * (KY / SLIDE_Y);
#endif
}

#include "all2all/gradient_descent/weights_update.cl"
