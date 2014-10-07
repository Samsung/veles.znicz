#include "gradient_descent_common.cl"


/// @brief Calculate gradient for bias update.
/// @param bias Layer bias.
/// @param err_output Backpropagated error.
/// @param gradient Computed gradient to store in if not null.
/// @param lr learning_rate.
/// @param factor_l12 lnorm_factor.
/// @param l1_vs_l2 how much to prefer l1 over l2 (from [0, 1]).
/// @param gradient_moment Moment for gradient.
/// @details gradient = previous_gradient * gradient_moment -
///                     lr * (sum(err_y) +
///                     factor_l12 * ((1 - l1_vs_l2) * bias + 0.5 * l1_vs_l2 * sign(bias)).
///          Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          BATCH - minibatch size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global const dtype    /* IN */    *err_output,
                 __global dtype     /* IN, OUT */    *bias,
                 __global dtype     /* IN, OUT */    *gradient,
                 const dtype             /* IN */    lr,
                 const dtype             /* IN */    factor_l12,
                 const dtype             /* IN */    l1_vs_l2,
                 const dtype             /* IN */    gradient_moment) {
 
  #define A err_output
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
    dtype gd = -lr * (sum + gradient_step_l12(cur_bias, factor_l12, l1_vs_l2));
    #if STORE_GRADIENT > 0
    gd += gradient[bx] * gradient_moment;
    gradient[bx] = gd;
    #endif
    #if APPLY_GRADIENT > 0
    bias[bx] = cur_bias + gd;
    #endif
  }
}
