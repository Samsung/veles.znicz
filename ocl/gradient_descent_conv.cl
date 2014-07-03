#include "defines.cl"
#include "highlight.cl"

#ifndef INCLUDE_BIAS
#error "INCLUDE_BIAS should be defined"
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

#ifdef USE_ATOMICS
/* @brief Kernels for convolutional layer gradient descent.
 * @details Should be defined externally:
 *          defines from conv_common.cl,
 *          REDUCE_SIZE - buffer size for reduce operation.
 */

/// @brief Computes backprogated error for previous layer.
/// @param err_y backpropagated error of the output layer.
/// @param weights weights.
/// @param err_h resulted backpropagated error for previous layer.
/// @details err_h = err_y * weights.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global const dtype    /* IN */    *err_y,
                  __global const dtype    /* IN */    *weights,
                  __global dtype         /* OUT */    *err_h) {

  #define A_WIDTH (BATCH * ((SX_FULL - KX) / SLIDE_X + 1) * ((SY_FULL - KY) / SLIDE_Y + 1))
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define AB_COMMON N_KERNELS

  #define A err_y
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
    ATOM_ADD(&err_h[IN_REAL_OFFS], sum);
  }
  #undef in_offs
}
#endif

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
  #define A h
  #define B err_y

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS
  #define A_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #else

  #define A_WIDTH N_KERNELS
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define A err_y
  #define B h

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


#if INCLUDE_BIAS > 0
/// @brief Calculate gradient for bias update.
/// @param bias Layer bias.
/// @param err_y Backpropagated error.
/// @param gradient Computed gradient to store in if not null.
/// @param alpha_batch (-global_alpha / batch_size).
/// @param alpha_lambda (-global_alpha * global_lambda).
/// @param gradient_moment Moment for gradient.
/// @details gradient = previous_gradient * gradient_moment +
///                     sum(err_y) * alpha_batch + bias * alpha_lambda.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global const dtype    /* IN */    *err_y,
                 __global dtype     /* IN, OUT */    *bias,
                 __global dtype     /* IN, OUT */    *gradient,
                 const dtype             /* IN */    alpha_batch,
                 const dtype             /* IN */    alpha_lambda,
                 const dtype             /* IN */    gradient_moment) {

  #define A err_y
  #define A_WIDTH N_KERNELS
  #define A_HEIGHT (BATCH * ((SX_FULL - KX) / SLIDE_X + 1) * ((SY_FULL - KY) / SLIDE_Y + 1))
  #define A_COL

  #include "matrix_reduce.cl"

  #undef A_COL
  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  if (!tx) {
    sum += AS[0];
    dtype cur_bias = bias[bx];
    dtype gd = sum * alpha_batch + cur_bias * alpha_lambda;
    #if STORE_GRADIENT > 0
    gd += gradient[bx] * gradient_moment;
    gradient[bx] = gd;
    #endif
    #if APPLY_GRADIENT > 0
    bias[bx] = cur_bias + gd;
    #endif
  }
}
#endif
#endif
