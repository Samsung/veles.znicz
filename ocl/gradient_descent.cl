#include "gradient_descent_common.cl"


/// @brief Computes backprogated error for previous layer:
///        err_h = err_y * weights.
/// @details Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global const dtype    /* IN */    *err_y,
                  __global const dtype    /* IN */    *weights,
                  __global dtype         /* OUT */    *err_h) {
  #define A_WIDTH BATCH
  #define B_WIDTH H
  #define AB_COMMON Y

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

  if (valid) {
    err_h[idx] = sum;
  }
}


/// @brief Calculate gradient for weights update.
/// @param err_y Backpropagated error.
/// @param h Layer input.
/// @param weights Layer weights.
/// @param gradient Computed gradient.
/// @param lr learning_rate.
/// @param lr_x_l learning_rate * lnorm_factor.
/// @param l1_vs_l2 how much to prefer l1 over l2 (from [0, 1]).
/// @param gradient_moment Moment for gradient.
/// @details gradient = previous_gradient * gradient_moment -
///                     (err_y * h * lr +
///                      lr_x_l * ((1 - l1_vs_l2) * weights + 0.5 * l1_vs_l2 * sign(weights)).
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global const dtype    /* IN */    *err_y,
                    __global const dtype    /* IN */    *h,
                    __global dtype     /* IN, OUT */    *weights,
                    __global dtype     /* IN, OUT */    *gradient,
                    const dtype             /* IN */    lr,
                    const dtype             /* IN */    lr_x_l,
                    const dtype             /* IN */    l1_vs_l2,
                    const dtype             /* IN */    gradient_moment) {
  #if WEIGHTS_TRANSPOSED > 0
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
    dtype weight = weights[idx];
    dtype gd = -gradient_step(weight, sum, lr, lr_x_l, l1_vs_l2);
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
/// @param lr learning_rate.
/// @param lr_x_l learning_rate * lnorm_factor.
/// @param l1_vs_l2 how much to prefer l1 over l2 (from [0, 1]).
/// @param gradient_moment Moment for gradient.
/// @details gradient = previous_gradient * gradient_moment -
///                     (sum(err_y) * lr +
///                      lr_x_l * ((1 - l1_vs_l2) * bias + 0.5 * l1_vs_l2 * sign(bias)).
///          Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          BATCH - minibatch size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global const dtype    /* IN */    *err_y,
                 __global dtype     /* IN, OUT */    *bias,
                 __global dtype     /* IN, OUT */    *gradient,
                 const dtype             /* IN */    lr,
                 const dtype             /* IN */    lr_x_l,
                 const dtype             /* IN */    l1_vs_l2,
                 const dtype             /* IN */    gradient_moment) {
 
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
    dtype cur_bias = bias[bx];
    dtype gd = -gradient_step(cur_bias, sum, lr, lr_x_l, l1_vs_l2);
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
