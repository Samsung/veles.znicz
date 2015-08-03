#include "defines.cl"
#include "highlight.cl"


#ifndef OUTPUT_SIZE
#error "OUTPUT_SIZE must be defined"
#endif


__kernel void add_forward(__global const dtype     *x,
						  __global const dtype     *y,
						  __global dtype           *output) {
  size_t index = get_global_id(0);
  output[index] = x[index] + y[index];
}
