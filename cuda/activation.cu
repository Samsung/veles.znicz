#include "defines.cu"
#include "highlight.cuh"


__global__ void forward_tanh(
    const dtype /* IN */*input, dtype /* OUT */*output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  output[idx] = tanh(input[idx] * (dtype) 0.6666) * (dtype) 1.7159;
}


__global__ void backward_tanh(const dtype    /* IN */    *input,
                              const dtype    /* IN */    *output,
                              const dtype    /* IN */    *err_output,
                              dtype         /* OUT */    *err_input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dtype vle = output[idx];
  err_input[idx] = err_output[idx] * (vle * vle * (dtype)(-0.388484177) + (dtype)1.14381894);
}


__global__ void forward_mul(const dtype    /* IN */    *input,
                            dtype         /* OUT */    *output,
                            const dtype    /* IN */    factor) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  output[idx] = input[idx] * factor;
}


__global__ void backward_mul(const dtype    /* IN */    *input,
                             const dtype    /* IN */    *output,
                             const dtype    /* IN */    *err_output,
                             dtype         /* OUT */    *err_input,
                             const dtype    /* IN */    factor) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  err_input[idx] = err_output[idx] * factor;
}


__global__ void forward_relu(const dtype    /* IN */    *input,
                             dtype         /* OUT */    *output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dtype vle = input[idx];
  output[idx] = vle > 15 ? vle : log(exp(vle) + 1);
}


__global__ void backward_relu(const dtype    /* IN */    *input,
                              const dtype    /* IN */    *output,
                              const dtype    /* IN */    *err_output,
                              dtype         /* OUT */    *err_input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  err_input[idx] = err_output[idx] * ((dtype)1.0 - exp(-output[idx]));
}


__global__ void forward_strict_relu(const dtype    /* IN */    *input,
                                    dtype         /* OUT */    *output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  output[idx] = max(input[idx], (dtype)0);
}


__global__ void backward_strict_relu(const dtype    /* IN */    *input,
                                     const dtype    /* IN */    *output,
                                     const dtype    /* IN */    *err_output,
                                     dtype         /* OUT */    *err_input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  err_input[idx] = output[idx] > 0 ? err_output[idx] : 0;
}


__global__ void forward_log(const dtype    /* IN */    *input,
                            dtype         /* OUT */    *output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dtype vle = input[idx];
  output[idx] = log(vle + sqrt(vle * vle + 1));
}


__global__ void backward_log(const dtype    /* IN */    *input,
                             const dtype    /* IN */    *output,
                             const dtype    /* IN */    *err_output,
                             dtype         /* OUT */    *err_input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dtype vle = input[idx];
  err_input[idx] = err_output[idx] * rsqrt(vle * vle + 1);
}


#define tanhlog_d 3
#define tanhlog_a ((dtype)0.242528761112)
#define tanhlog_b ((dtype)305.459953195)

__global__ void forward_tanhlog(const dtype    /* IN */    *input,
                                dtype         /* OUT */    *output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
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

__global__ void backward_tanhlog(const dtype    /* IN */    *input,
                                 const dtype    /* IN */    *output,
                                 const dtype    /* IN */    *err_output,
                                 dtype         /* OUT */    *err_input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
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


__global__ void forward_sincos(const dtype    /* IN */    *input,
                               dtype         /* OUT */    *output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dtype vle = input[idx];
  output[idx] = (idx & 1) ? sin(vle) : cos(vle);
}


__global__ void backward_sincos(const dtype    /* IN */    *input,
                                const dtype    /* IN */    *output,
                                const dtype    /* IN */    *err_output,
                                dtype         /* OUT */    *err_input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  dtype vle = input[idx];
  err_input[idx] = err_output[idx] * ((idx & 1) ? cos(vle) : -sin(vle));
}
