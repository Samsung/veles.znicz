#include "defines.cl"
#include "highlight.cl"


__kernel void forward_tanh(__global const dtype    /* IN */    *input,
                           __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  output[idx] = tanh(input[idx] * (dtype)0.6666) * (dtype)1.7159;
}


__kernel void backward_tanh(__global const dtype    /* IN */    *input,
                            __global const dtype    /* IN */    *output,
                            __global const dtype    /* IN */    *err_output,
                            __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  dtype vle = output[idx];
  err_input[idx] = err_output[idx] * (vle * vle * (dtype)(-0.388484177) + (dtype)1.14381894);
}


__kernel void forward_sigmoid(__global const dtype    /* IN */    *input,
                              __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}


__kernel void backward_sigmoid(__global const dtype    /* IN */    *input,
                               __global const dtype    /* IN */    *output,
                               __global const dtype    /* IN */    *err_output,
                               __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  dtype vle = output[idx];
  err_input[idx] = err_output[idx] * (vle * ((dtype)1.0 - vle));
}


__kernel void forward_mul(__global const dtype    /* IN */    *input,
                          __global dtype         /* OUT */    *output,
                          const dtype             /* IN */    factor) {
  int idx = get_global_id(0);
  output[idx] = input[idx] * factor;
}


__kernel void backward_mul(__global const dtype    /* IN */    *input,
                           __global const dtype    /* IN */    *output,
                           __global const dtype    /* IN */    *err_output,
                           __global dtype         /* OUT */    *err_input,
                           const dtype             /* IN */    factor) {
  int idx = get_global_id(0);
  err_input[idx] = err_output[idx] * factor;
}


__kernel void forward_relu(__global const dtype    /* IN */    *input,
                           __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  dtype vle = input[idx];
  output[idx] = vle > 15 ? vle : log(exp(vle) + 1);
}


__kernel void backward_relu(__global const dtype    /* IN */    *input,
                            __global const dtype    /* IN */    *output,
                            __global const dtype    /* IN */    *err_output,
                            __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  err_input[idx] = err_output[idx] * ((dtype)1.0 - exp(-output[idx]));
}


__kernel void forward_strict_relu(__global const dtype    /* IN */    *input,
                                  __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  output[idx] = max(input[idx], (dtype)0);
}


__kernel void backward_strict_relu(__global const dtype    /* IN */    *input,
                                   __global const dtype    /* IN */    *output,
                                   __global const dtype    /* IN */    *err_output,
                                   __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  err_input[idx] = output[idx] > 0 ? err_output[idx] : 0;
}


__kernel void forward_log(__global const dtype    /* IN */    *input,
                          __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  dtype vle = input[idx];
  output[idx] = log(vle + sqrt(vle * vle + 1));
}


__kernel void backward_log(__global const dtype    /* IN */    *input,
                           __global const dtype    /* IN */    *output,
                           __global const dtype    /* IN */    *err_output,
                           __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  dtype vle = input[idx];
  err_input[idx] = err_output[idx] * rsqrt(vle * vle + 1);
}


#define tanhlog_d 3
#define tanhlog_a ((dtype)0.242528761112)
#define tanhlog_b ((dtype)305.459953195)

__kernel void forward_tanhlog(__global const dtype    /* IN */    *input,
                              __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  dtype x = input[idx];
  dtype xx = fabs(x);
  dtype y;
  if (xx <= tanhlog_d) {
    y = tanh(x * (dtype)0.6666) * (dtype)1.7159;
  }
  else {
    y = copysign(log(xx * tanhlog_b) * tanhlog_a, x);
  }
  output[idx] = y;
}

__kernel void backward_tanhlog(__global const dtype    /* IN */    *input,
                               __global const dtype    /* IN */    *output,
                               __global const dtype    /* IN */    *err_output,
                               __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  dtype x = input[idx];
  dtype xx = fabs(x);
  dtype y;
  if (xx <= tanhlog_d) {
    y = output[idx];
    y = y * y * (-0.388484177) + (dtype)1.14381894;
  }
  else {
    y = fabs(tanhlog_a / x);
  }
  err_input[idx] = err_output[idx] * y;
}


__kernel void forward_sincos(__global const dtype    /* IN */    *input,
                             __global dtype         /* OUT */    *output) {
  int idx = get_global_id(0);
  dtype vle = input[idx];
  output[idx] = (idx & 1) ? sin(vle) : cos(vle);
}


__kernel void backward_sincos(__global const dtype    /* IN */    *input,
                              __global const dtype    /* IN */    *output,
                              __global const dtype    /* IN */    *err_output,
                              __global dtype         /* OUT */    *err_input) {
  int idx = get_global_id(0);
  dtype vle = input[idx];
  err_input[idx] = err_output[idx] * ((idx & 1) ? cos(vle) : -sin(vle));
}
