#include "defines.cu"
#include "highlight.cuh"
#include "random.cu"

/// @brief xorshift128+
extern "C"
__global__ void dropout_forward(const dtype /* IN */ *inputs,
                                const ulong /* IN */ threshold,
                                const dtype /* IN */ pass,
                                      ulong2    /* IN, OUT */    *states,
                                      dtype         /* OUT */    *mask,
                                      dtype         /* OUT */    *output) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  ulong random;
  xorshift128plus(states[index], random);
  dtype val = random < threshold ? 0 : pass;
  mask[index] = val;
  output[index] = inputs[index] * val;
}

__global__ void dropout_backward(const dtype *mask,
                                 const dtype *err_y,
                                 dtype *err_h) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  err_h[idx] = err_y[idx] * mask[idx];
}
