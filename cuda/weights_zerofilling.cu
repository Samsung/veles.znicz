#include "defines.cu"

/// @brief Multiplies weights with given mask. Dimensions should be same.

extern "C" __global__ void multiply_by_mask(dtype* mask, dtype* weights) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  weights[idx] *= mask[idx];
}
