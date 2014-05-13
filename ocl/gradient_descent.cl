#include "defines.cl"
#include "highlight.cl"


/// @brief Computes backprogated error for previous layer:
///        err_h = err_y * weights.
/// @details Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global const c_dtype    /* IN */    *err_y,
                  __global const c_dtype    /* IN */    *weights,
                  __global c_dtype         /* OUT */    *err_h) {
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

  if (valid) {
    err_h[idx] = sum[0];
  }
}


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
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global const c_dtype    /* IN */    *err_y,
                    __global const c_dtype    /* IN */    *h,
                    __global c_dtype     /* IN, OUT */    *weights,
                    __global c_dtype     /* IN, OUT */    *gradient,
                    const dtype               /* IN */    alpha_batch,
                    const dtype               /* IN */    alpha_lambda,
                    const dtype               /* IN */    gradient_moment) {
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
    gd += gradient[idx] * gradient_moment;
    gradient[idx] = gd;
    #endif
    #ifdef APPLY_GRADIENT
    weights[idx] = weight + gd;
    #endif
  }
}


/// @brief Calculate gradient for bias update.
/// @param bias Layer bias.
/// @param err_y Backpropagated error.
/// @param gradient Computed gradient to store in if not null.
/// @param alpha_batch (-global_alpha / batch_size).
/// @details gradient = previous_gradient * gradient_moment +
///                     sum(err_y) * alpha_batch.
///          Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          BATCH - minibatch size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global const c_dtype    /* IN */    *err_y,
                 __global c_dtype     /* IN, OUT */    *bias,
                 __global c_dtype     /* IN, OUT */    *gradient,
                 const dtype               /* IN */    alpha_batch,
                 const dtype               /* IN */    gradient_moment) {
 
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
    gd += gradient[bx] * gradient_moment;
    gradient[bx] = gd;
    #endif
    #ifdef APPLY_GRADIENT
    bias[bx] += gd;
    #endif
  }
}
