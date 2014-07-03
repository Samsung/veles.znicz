#include "defines.cl"
#include "highlight.cl"
#include "random.cl"

/// @brief xorshift128+
__kernel void dropout_forward(__global const dtype    /* IN */    *inputs,
                              const ulong             /* IN */    threshold,
                              const dtype             /* IN */    pass,
                              __global ulong2    /* IN, OUT */    *states,
                              __global dtype         /* OUT */    *mask,
                              __global dtype         /* OUT */    *output) {
  int index = get_global_id(0);
  ulong random;
  xorshift128plus(states[index], random);
  dtype val = random < threshold ? 0 : pass;
  mask[index] = val;
  output[index] = inputs[index] * val;
}

__kernel void dropout_backward(__global const dtype *mask,
                               __global const dtype *err_y,
                               __global dtype *err_h) {
  int idx = get_global_id(0);
  err_h[idx] = err_y[idx] * mask[idx];
}
