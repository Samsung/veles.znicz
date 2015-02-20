#include "defines.cu"


/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= y * (1 - y)
extern "C"
__global__ void err_y_update(dtype *err_y, const dtype *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ERR_OUTPUT_SIZE) {
    dtype x = y[idx];
    err_y[idx] *= x * ((dtype)1.0 - x);
  }
}
