#include "defines.cu"
#include "highlight.cu"


#ifndef OUTPUT_SIZE
#error "OUTPUT_SIZE must be defined"
#endif


extern "C"
__global__ void add_forward(const dtype     *x,
                            const dtype     *y,
                            dtype           *output) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < OUTPUT_SIZE) {
	  output[index] = x[index] + y[index];
  }
}
