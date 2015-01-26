#include "defines.cu"

extern "C"
__global__ void feed_layer(const dtype    *input,
                           const int      *output_offset,
                           dtype          *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < INPUT_SIZE) {
    output[output_offset[idx]] = input[idx];
  }
}
