#include "defines.cu"
#include "highlight.cu"
#include "random.cu"


extern "C"
__global__ void dropout_forward(const dtype     *inputs,
                                const ulong     threshold,
                                const dtype     pass,
                                ulonglong2      *states,
                                dtype           *mask,
                                dtype           *output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < OUTPUT_SIZE) {
    ulong random;
    xorshift128plus(states[index], random);
    dtype val = random < threshold ? 0 : pass;
    mask[index] = val;
    output[index] = inputs[index] * val;
  }
}


extern "C"
__global__ void dropout_backward(const dtype *mask,
                                 const dtype *err_y,
                                 dtype *err_h) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < OUTPUT_SIZE) {
    err_h[index] = err_y[index] * mask[index];
  }
}
