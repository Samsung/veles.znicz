#include "defines.cu"


/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= 1.0 - exp(-y)
extern "C"
__global__ void err_y_update(dtype *err_y, const dtype *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ERR_OUTPUT_SIZE) {
    err_y[idx] *= (dtype)1.0 - exp(-y[idx]);
  }
}
