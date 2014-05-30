#include "defines.cl"
#include "highlight.cl"


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
  err_input[idx] = err_output[idx] * c_log_back(vle);
}
