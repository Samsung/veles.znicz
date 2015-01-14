#include "defines.cu"


/// @brief Updates backpropagated error by activation derivative.
/// @details err_y *= y * y * (-0.388484177) + 1.14381894
extern "C"
__global__ void err_y_update(dtype *err_y, const dtype *y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (Y * BATCH)) {
    dtype x = y[idx];
    err_y[idx] *= x * x * (dtype)(-0.388484177) + (dtype)1.14381894;
  }
}
