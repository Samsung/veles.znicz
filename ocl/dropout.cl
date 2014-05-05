#include "defines.cl"
#include "highlight.cl"

/// @brief xorshift128+
__kernel void dropout_forward(__global const c_dtype /* IN */    *inputs,
                              const ulong            /* IN */    threshold,
                              const c_dtype          /* IN */    pass,
                              __global ulong2   /* IN, OUT */    *states,
                              __global c_dtype      /* OUT */    *mask,
                              __global c_dtype      /* OUT */    *output) {
  int index = get_global_id(0);

  ulong2 seed = states[index];
  ulong s1 = seed.x;
  const ulong s0 = seed.y;
  seed.x = s0;
  s1 ^= s1 << 23;
  ulong random = (seed.y = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
  states[index] = seed;

  c_dtype val = random < threshold ? c_from_re(0) : pass;
  mask[index] = val;
  output[index] = inputs[index] * val;
}

__kernel void dropout_backward(__global const c_dtype *mask,
                               __global c_dtype *err_y) {
  int idx = get_global_id(0);
  err_y[idx] *= mask[idx];
}
