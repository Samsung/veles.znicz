#include "defines.cl"
#include "highlight.cl"

/// @brief Computes backprogated error for previous layer:
///        err_h = err_y * weights
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @details Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IN*/ *weights,
                  __global c_dtype /*OUT*/ *err_h) {
  #define A_WIDTH BATCH
  #define B_WIDTH H
  #define AB_COMMON Y

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

  if (valid)
    err_h[idx] = sum[0];
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
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IN*/  *h,
                    __global c_dtype /*IO*/ *weights, __global c_dtype /*OUT*/ *gradient,
                    const dtype alpha_batch, const dtype alpha_lambda) {
  #ifdef WEIGHTS_TRANSPOSED
  #define A_WIDTH H
  #define B_WIDTH Y
  #define A h
  #define B err_y
  #else
  #define A_WIDTH Y
  #define B_WIDTH H
  #define A err_y
  #define B h
  #endif

  #define AB_COMMON BATCH

  #define A_COL
  #define B_COL

  #include "matrix_multiplication.cl"

  #undef A_COL
  #undef B_COL

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

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
///          Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          BATCH - minibatch size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IO*/ *bias,
                 __global c_dtype /*OUT*/ *gradient, const dtype alpha_batch) {
 
  #define A err_y
  #define A_WIDTH Y
  #define A_HEIGHT BATCH
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
