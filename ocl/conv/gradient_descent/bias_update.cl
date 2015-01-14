#include "gradient_descent_common.cl"
#include "conv_common.cl"


/// @brief Calculate gradient for bias update.
/// @details See gradient_descent.cl.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global const dtype    /* IN */    *err_output,
                 __global dtype     /* IN, OUT */    *bias,
                 __global dtype     /* IN, OUT */    *gradient,
                 __global dtype     /* IN, OUT */    *accumulated_gradient,
                 __global dtype     /* IN, OUT */    *gradient_with_moment,
                 const dtype             /* IN */    lr,
                 const dtype             /* IN */    factor_l12,
                 const dtype             /* IN */    l1_vs_l2,
                 const dtype             /* IN */    moment) {

  #define A err_output
  #define A_WIDTH N_KERNELS
  #define A_HEIGHT (BATCH * KERNELS_PER_SAMPLE)
  #define A_COL

  #include "matrix_reduce.cl"

  #undef A_COL
  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  #include "bias_update.store_output.cl"
}
