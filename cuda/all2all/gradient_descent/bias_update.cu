#include "gradient_descent_common.cu"


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
///          BIAS_SIZE - bias size (Y),
///          OUTPUT_SIZE - number of output elements in the minibatch (BATCH).
extern "C"
__global__ void bias_update(const dtype    *err_output,
                            dtype          *bias,
                            dtype          *gradient,
                            dtype          *accumulated_gradient,
                            dtype          *gradient_with_moment,
                            const dtype    lr,
                            const dtype    factor_l12,
                            const dtype    l1_vs_l2,
                            const dtype    moment) {

  #define A err_output
  #define A_WIDTH BIAS_SIZE
  #define A_HEIGHT OUTPUT_SIZE
  #define A_COL

  #include "matrix_reduce.cu"

  #undef A_COL
  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  #include "bias_update.store_output.cu"
}
