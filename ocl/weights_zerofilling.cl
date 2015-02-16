#include "defines.cl"
#include "highlight.cl"

/// @brief Multiplies weights with given mask. Dimensions should be same.

__kernel void multiply_by_mask(__global dtype* mask,
                               __global dtype* weights) {
  size_t idx = get_global_id(0);
  weights[idx] *= mask[idx];
}
