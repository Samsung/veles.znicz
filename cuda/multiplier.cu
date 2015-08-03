#include "defines.cu"
#include "highlight.cu"


#ifndef OUTPUT_SIZE
#error "OUTPUT_SIZE must be defined"
#endif


extern "C"
__global__ void multiply_forward(const dtype     *x,
                                 const dtype     *y,
                                 dtype           *output) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < OUTPUT_SIZE) {
	  output[index] = x[index] * y[index];
  }
}


extern "C"
__global__ void multiply_backward(const dtype     *x,
                                  const dtype     *y,
								  const dtype     *err_output,
                                  dtype           *err_x,
								  dtype           *err_y) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < OUTPUT_SIZE) {
	  dtype err = err_output[index];
	  dtype out_x = err * y[index];
	  dtype out_y = err * x[index];
	  err_x[index] = out_x;
	  err_y[index] = out_y;
  }
}
