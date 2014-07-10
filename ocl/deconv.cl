#include "defines.cl"
#include "highlight.cl"

#ifndef INCLUDE_BIAS
#error "INCLUDE_BIAS should be defined"
#endif
#if INCLUDE_BIAS != 0
#error "INCLUDE_BIAS should be 0"
#endif

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

#ifndef STORE_GRADIENT
#error "STORE_GRADIENT should be defined"
#endif

#ifndef APPLY_GRADIENT
#error "APPLY_GRADIENT should be defined"
#endif

#include "conv_common.cl"

/// @brief Does deconvolution.
/// @param input output of the corresponding convolutional layer.
/// @param weights weights.
/// @param output deconvolution of input.
/// @param hits number of the summations to this point of output.
/// @details output = input * weights.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global const dtype      /* IN */    *input,
                __global const dtype      /* IN */    *weights,
                __global dtype           /* OUT */    *output,
                __global volatile int    /* OUT */    *hits) {

  #define A_WIDTH (BATCH * ((SX_FULL - KX) / SLIDE_X + 1) * ((SY_FULL - KY) / SLIDE_Y + 1))
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define AB_COMMON N_KERNELS

  #define A input
  #define B weights

  #if WEIGHTS_TRANSPOSED <= 0
  #define B_COL
  #endif

  #include "matrix_multiplication.cl"

  #if WEIGHTS_TRANSPOSED <= 0
  #undef B_COL
  #endif

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  #define in_offs idx
  if ((valid) && (IN_REAL_OFFS_VALID)) {
    ATOM_ADD(&output[IN_REAL_OFFS], sum);
    atomic_inc(&hits[IN_REAL_OFFS]);
  }
  #undef in_offs
}


__kernel
void apply_hits(__global dtype    /* IN, OUT */    *output,
                __global const int     /* IN */    *hits) {
  int idx = get_global_id(0);
  int n = hits[idx];
  output[idx] /= n ? n : 1;
}

KERNEL_CLEAR(clear_hits, int)

KERNEL_CLEAR(clear_output, dtype)
