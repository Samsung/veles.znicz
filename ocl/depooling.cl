#include "defines.cl"
#include "highlight.cl"

__kernel void feed_layer(__global const dtype    /* IN */    *input,
                         __global const int      /* IN */    *output_offset,
                         __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  output[output_offset[idx]] = input[idx];
}

KERNEL_CLEAR(output_clear, dtype)
