#include "gradient_descent_common.cl"


/// @brief Calculate gradient for bias update.
/// @param bias Layer bias.
/// @param err_output Backpropagated error.
/// @param gradient Computed gradient.
/// @param gradient_with_moment Accumulated gradient with moments.
/// @param lr learning_rate.
/// @param factor_l12 lnorm_factor.
/// @param l1_vs_l2 how much to prefer l1 over l2 (from [0, 1]).
/// @param moment Moment for gradient.
/// @details Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          BATCH - minibatch size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global const dtype    /* IN */    *err_output,
                 __global dtype     /* IN, OUT */    *bias,
                 __global dtype     /* IN, OUT */    *gradient,
                 __global dtype     /* IN, OUT */    *gradient_with_moment,
                 const dtype             /* IN */    lr,
                 const dtype             /* IN */    factor_l12,
                 const dtype             /* IN */    l1_vs_l2,
                 const dtype             /* IN */    moment) {
 
  #define A err_output
  #define A_WIDTH Y
  #define A_HEIGHT BATCH
  #define A_COL

  #include "matrix_reduce.cl"

  #undef A_COL
  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  #include "bias_update.store_output.cl"
}
