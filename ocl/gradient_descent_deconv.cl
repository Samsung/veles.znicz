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

#include "conv.cl"

#if (STORE_GRADIENT > 0) || (APPLY_GRADIENT > 0)
/// @brief Calculate gradient for weights update.
/// @param err_y Backpropagated error.
/// @param h Layer input.
/// @param weights Layer weights.
/// @param gradient Computed gradient.
/// @param alpha_batch (-global_alpha / batch_size).
/// @param alpha_lambda (-global_alpha * global_lambda).
/// @param gradient_moment Moment for gradient.
/// @details gradient = previous_gradient * gradient_moment +
///                     err_y * h * alpha_batch + weights * alpha_lambda.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global const dtype    /* IN */    *err_y,
                    __global const dtype    /* IN */    *h,
                    __global dtype     /* IN, OUT */    *weights,
                    __global dtype     /* IN, OUT */    *gradient,
                    const dtype             /* IN */    alpha_batch,
                    const dtype             /* IN */    alpha_lambda,
                    const dtype             /* IN */    gradient_moment) {
  #if WEIGHTS_TRANSPOSED > 0

  #define A_WIDTH ELEMENTS_PER_KERNEL
  #define B_WIDTH N_KERNELS
  #define A err_y
  #define B h

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS
  #define A_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #else

  #define A_WIDTH N_KERNELS
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define A h
  #define B err_y

  #define in_offs b_offs
  #define B_REAL_OFFS IN_REAL_OFFS
  #define B_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #endif

  #define AB_COMMON (BATCH * KERNELS_PER_SAMPLE)

  #define A_COL
  #define B_COL

  #include "matrix_multiplication.cl"

  #undef A_COL
  #undef B_COL

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON
  #undef in_offs

  #undef A
  #undef B

  if (valid) {
    dtype weight = weights[idx];
    dtype gd = sum * alpha_batch + weight * alpha_lambda;
    #if STORE_GRADIENT > 0
    gd += gradient[idx] * gradient_moment;
    gradient[idx] = gd;
    #endif
    #if APPLY_GRADIENT > 0
    weights[idx] = weight + gd;
    #endif
  }
}
#endif

__kernel
void apply_hits(__global dtype    /* IN, OUT */    *err_output,
                __global const int     /* IN */    *hits) {
  int idx = get_global_id(0);
  int n = hits[idx];
  err_output[idx] /= n ? n : 1;
}
