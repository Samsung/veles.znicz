#include "gradient_descent_common.cu"


#if USE_ORTHO > 0
#include "weights_ortho.cu"
#endif


extern "C"
__global__ void weights_update(dtype          *weights,
                               const dtype    *gradient,
                               dtype          *accumulated_gradient,
                               dtype          *gradient_with_moment,
                               const dtype    lr,
                               const dtype    factor_l12,
                               const dtype    l1_vs_l2,
                               const dtype    moment,
                               const dtype    acc_alpha,
                               const dtype    acc_beta,
                               const dtype    gd_alpha,
                               const dtype    gd_beta
#if USE_ORTHO > 0
                             , const dtype    factor_ortho,
                               const dtype    *col_sums
#endif
                    ) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (H * Y)) {
    dtype sum = gradient[idx];
    #include "weights_update.store_output.cu"
  }
}
