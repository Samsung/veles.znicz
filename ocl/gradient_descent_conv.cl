#include "defines.cl"

/* @brief Kernels for convolutional layer gradient descent.
 * @author Kazantsev Alexey <a.kazantsev@samsung.com>
 * @details Should be defined externally:
 *          BLOCK_SIZE - size of the block for matrix multiplication,
 *          BATCH - minibatch size,
 *          SX - input image width,
 *          SY - input image height,
 *          N_CHANNELS - number of input channels,
 *          KX - kernel width,
 *          KY - kernel height,
 *          N_KERNELS - number of kernels (i.e. neurons),
 *          PAD_TOP - padding-top,
 *          PAD_LEFT - padding-left,
 *          PAD_BOTTOM - padding-bottom,
 *          PAD_RIGHT - padding-right,
 *          SLIDE_X - kernel sliding by x-axis,
 *          SLIDE_Y - kernel sliding by y-axis,
 *          REDUCE_SIZE - buffer size for reduce operation.
 */

#define SX_FULL (SX + PAD_LEFT + PAD_RIGHT)
#define SY_FULL (SY + PAD_TOP + PAD_BOTTOM)

#define KERNEL_APPLIES_PER_WIDTH ((SX_FULL - KX) / SLIDE_X + 1)
#define KERNEL_APPLIES_PER_HEIGHT ((SY_FULL - KY) / SLIDE_Y + 1)
#define KERNELS_PER_SAMPLE (KERNEL_APPLIES_PER_WIDTH * KERNEL_APPLIES_PER_HEIGHT)

#define ELEMENTS_PER_KERNEL (N_CHANNELS * KX * KY)
#define KERNEL_APPLY_NUMBER (in_offs / ELEMENTS_PER_KERNEL)
#define OFFS_IN_KERNEL (in_offs % ELEMENTS_PER_KERNEL)
#define PLAIN_ROW_IN_KERNEL (OFFS_IN_KERNEL / (N_CHANNELS * KX))
#define PLAIN_COL_IN_KERNEL (OFFS_IN_KERNEL % (N_CHANNELS * KX))
#define SAMPLE_NUMBER (KERNEL_APPLY_NUMBER / KERNELS_PER_SAMPLE)
#define KERNEL_APPLY_IN_SAMPLE (KERNEL_APPLY_NUMBER % KERNELS_PER_SAMPLE)
#define VIRT_ROW_IN_SAMPLE (KERNEL_APPLY_IN_SAMPLE / KERNEL_APPLIES_PER_WIDTH)
#define VIRT_COL_IN_SAMPLE (KERNEL_APPLY_IN_SAMPLE % KERNEL_APPLIES_PER_WIDTH)

#define SAMPLE_ROW (VIRT_ROW_IN_SAMPLE * SLIDE_Y + PLAIN_ROW_IN_KERNEL - PAD_TOP)
#define SAMPLE_COL (VIRT_COL_IN_SAMPLE * SLIDE_X * N_CHANNELS + PLAIN_COL_IN_KERNEL - PAD_LEFT * N_CHANNELS)

#define IN_REAL_OFFS_VALID ((SAMPLE_ROW >= 0) && (SAMPLE_ROW < SY) && (SAMPLE_COL >= 0) && (SAMPLE_COL < SX * N_CHANNELS))
#define IN_REAL_OFFS ((SAMPLE_NUMBER * SY + SAMPLE_ROW) * (SX * N_CHANNELS) + SAMPLE_COL)


/// @brief Computes backprogated error for previous layer.
/// @param err_y backpropagated error of the output layer.
/// @param weights weights.
/// @param err_h resulted backpropagated error for previous layer.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @details err_h_tmp = err_y * weights
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IN*/ *weights,
                  __global c_dtype /*OUT*/ *err_h) {

  #define A_WIDTH (BATCH * ((SX_FULL - KX) / SLIDE_X + 1) * ((SY_FULL - KY) / SLIDE_Y + 1))
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define AB_COMMON N_KERNELS

  #define A err_y
  #define B weights

  #ifndef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #include "matrix_multiplication.cl"

  #ifndef WEIGHTS_TRANSPOSED
  #undef B_COL
  #endif

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  #define in_offs idx
  if ((valid) && (IN_REAL_OFFS_VALID)) {
    ATOM_ADD(&err_h[IN_REAL_OFFS], sum[0]);
  }
  #undef in_offs
}


/// @brief Calculate gradient for weights update.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param err_y backpropagated error
/// @param h layer input
/// @param weights layer weights
/// @param gradient computed gradient to store in if not null
/// @param alpha_batch (-global_alpha / batch_size)
/// @param alpha_lambda (-global_alpha * global_lambda)
/// @details gradient = err_y * h * alpha_batch + weights * alpha_lambda
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IN*/  *h,
                    __global c_dtype /*IO*/ *weights, __global c_dtype /*OUT*/ *gradient,
                    const dtype alpha_batch, const dtype alpha_lambda) {
  #ifdef WEIGHTS_TRANSPOSED

  #define A_WIDTH ELEMENTS_PER_KERNEL
  #define B_WIDTH N_KERNELS
  #define A h
  #define B err_y

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS

  #else

  #define A_WIDTH N_KERNELS
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define A err_y
  #define B h

  #define in_offs b_offs
  #define B_REAL_OFFS IN_REAL_OFFS

  #endif

  #define AB_COMMON (BATCH * ((SX_FULL - KX) / SLIDE_X + 1) * ((SY_FULL - KY) / SLIDE_Y + 1))

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
    c_dtype weight = weights[idx];
    c_dtype gd = sum[0] * alpha_batch + weight * alpha_lambda;
    #ifdef STORE_GRADIENT
    gradient[idx] += gd;
    #endif
    #ifdef APPLY_GRADIENT
    weights[idx] = weight + gd;
    #endif
  }
}


/// @brief Calculate gradient for bias update.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param bias layer bias
/// @param err_y backpropagated error
/// @param gradient computed gradient to store in if not null
/// @param alpha_batch (-global_alpha / batch_size)
/// @details gradient = sum(err_y) * alpha_batch
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IO*/ *bias,
                 __global c_dtype /*OUT*/ *gradient, const dtype alpha_batch) {

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

    c_dtype gd = sum * alpha_batch;
    #ifdef STORE_GRADIENT
    gradient[bx] += gd;
    #endif
    #ifdef APPLY_GRADIENT
    bias[bx] += gd;
    #endif
  }
}
