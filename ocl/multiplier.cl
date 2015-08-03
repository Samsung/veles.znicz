#include "defines.cl"
#include "highlight.cl"


#ifndef OUTPUT_SIZE
#error "OUTPUT_SIZE must be defined"
#endif


__kernel void multiply_forward(__global const dtype     *x,
                               __global const dtype     *y,
                               __global dtype           *output) {
  size_t index = get_global_id(0);
  output[index] = x[index] * y[index];
}


__kernel void multiply_backward(__global const dtype     *x,
								__global const dtype     *y,
								__global const dtype     *err_output,
								__global dtype           *err_x,
								__global dtype           *err_y) {
  size_t index = get_global_id(0);
  dtype err = err_output[index];
  dtype out_x = err * y[index];
  dtype out_y = err * x[index];
  err_x[index] = out_x;
  err_y[index] = out_y;
}
