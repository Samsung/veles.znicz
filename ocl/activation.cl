#include "defines.cl"
#include "highlight.cl"


__kernel void forward_tanh(__global const c_dtype    /* IN */    *input,
                           __global c_dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  output[idx] = (dtype)1.7159 * c_tanh(input[idx] * (dtype)0.6666);
}


__kernel void backward_tanh(__global const c_dtype    /* IN */    *input,
                            __global const c_dtype    /* IN */    *output,
                            __global const c_dtype    /* IN */    *err_output,
                            __global c_dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  c_dtype vle = output[idx];
  err_input[idx] = c_mul(err_output[idx], c_mul(vle, vle) * (dtype)(-0.388484177) + (dtype)1.14381894);
}


__kernel void forward_strict_relu(__global const c_dtype    /* IN */    *input,
                                  __global c_dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  c_dtype vle = input[idx];
  c_re(vle) = max(c_re(vle), (dtype)0);
  output[idx] = vle;
}


__kernel void backward_strict_relu(__global const c_dtype    /* IN */    *input,
                                   __global const c_dtype    /* IN */    *output,
                                   __global const c_dtype    /* IN */    *err_output,
                                   __global c_dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  err_input[idx] = c_re(output[idx]) > 0 ? err_output[idx] : c_from_re(0);
}


__kernel void forward_log(__global const c_dtype    /* IN */    *input,
                          __global c_dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  c_dtype vle = input[idx];
  output[idx] = c_log_act(vle);
}


__kernel void backward_log(__global const c_dtype    /* IN */    *input,
                           __global const c_dtype    /* IN */    *output,
                           __global const c_dtype    /* IN */    *err_output,
                           __global c_dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  c_dtype vle = input[idx];
  err_input[idx] = c_mul(err_output[idx], c_log_back(vle));
}


__kernel void forward_sincos(__global const c_dtype    /* IN */    *input,
                             __global c_dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  c_dtype vle = input[idx];
  output[idx] = (idx & 1) ? c_sin(vle) : c_cos(vle);
}


__kernel void backward_sincos(__global const c_dtype    /* IN */    *input,
                              __global const c_dtype    /* IN */    *output,
                              __global const c_dtype    /* IN */    *err_output,
                              __global c_dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  c_dtype vle = input[idx];
  err_input[idx] = c_mul(err_output[idx], ((idx & 1) ? c_cos(vle) : -c_sin(vle)));
}
