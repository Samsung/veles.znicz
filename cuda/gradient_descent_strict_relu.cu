#include "defines.cu"


/// Strict ReLU back propagation
/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= (y > 0) ? 1 : 0
extern "C"
__global__ void err_y_update(dtype *err_y, const dtype *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ((idx < ERR_OUTPUT_SIZE) && (y[idx] <= 0)) {
    err_y[idx] = 0;
  }
}
