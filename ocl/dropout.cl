#include "defines.cl"
#include "highlight.cl"

/// @brief xorshift128+
__kernel void dropout_forward(__global const dtype    /* IN */    *inputs,
                              const ulong             /* IN */    threshold,
                              const dtype             /* IN */    pass,
                              __global ulong2    /* IN, OUT */    *states,
                              __global dtype         /* OUT */    *mask,
                              __global dtype         /* OUT */    *output) {
  int index = get_global_id(0);

  ulong2 seed = states[index];
  ulong s1 = seed.x;
  const ulong s0 = seed.y;
  seed.x = s0;
  s1 ^= s1 << 23;
  ulong random = (seed.y = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
  states[index] = seed;

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
