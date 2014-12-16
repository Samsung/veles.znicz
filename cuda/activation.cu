#include "defines.cu"


extern "C"
__global__ void forward_tanh(const dtype    /* IN */    *input,
                             dtype         /* OUT */    *output) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  output[idx] = tanh(input[idx] * (dtype)0.6666) * (dtype)1.7159;
}


extern "C"
__global__ void backward_tanh(const dtype    /* IN */    *input,
                              const dtype    /* IN */    *output,
                              const dtype    /* IN */    *err_output,
                              dtype         /* OUT */    *err_input) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype vle = output[idx];
  err_input[idx] = err_output[idx] * (vle * vle * (dtype)(-0.388484177) + (dtype)1.14381894);
}


extern "C"
__global__ void forward_mul(const dtype    /* IN */    *input,
                            dtype         /* OUT */    *output,
                            const dtype    /* IN */    factor) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  output[idx] = input[idx] * factor;
}


extern "C"
__global__ void backward_mul(const dtype    /* IN */    *input,
                             const dtype    /* IN */    *output,
                             const dtype    /* IN */    *err_output,
                             dtype         /* OUT */    *err_input,
                             const dtype    /* IN */    factor) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  err_input[idx] = err_output[idx] * factor;
}


extern "C"
__global__ void forward_relu(const dtype    /* IN */    *input,
                             dtype         /* OUT */    *output) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype vle = input[idx];
  output[idx] = vle > 15 ? vle : log(exp(vle) + 1);
}


extern "C"
__global__ void backward_relu(const dtype    /* IN */    *input,
                              const dtype    /* IN */    *output,
                              const dtype    /* IN */    *err_output,
                              dtype         /* OUT */    *err_input) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  err_input[idx] = err_output[idx] * ((dtype)1.0 - exp(-output[idx]));
}


extern "C"
__global__ void forward_strict_relu(const dtype    /* IN */    *input,
                                    dtype         /* OUT */    *output) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  output[idx] = max(input[idx], (dtype)0);
}


extern "C"
__global__ void backward_strict_relu(const dtype    /* IN */    *input,
                                     const dtype    /* IN */    *output,
                                     const dtype    /* IN */    *err_output,
                                     dtype         /* OUT */    *err_input) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  err_input[idx] = output[idx] > 0 ? err_output[idx] : 0;
}


extern "C"
__global__ void forward_log(const dtype    /* IN */    *input,
                            dtype         /* OUT */    *output) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype vle = input[idx];
  output[idx] = log(vle + sqrt(vle * vle + 1));
}


extern "C"
__global__ void backward_log(const dtype    /* IN */    *input,
                             const dtype    /* IN */    *output,
                             const dtype    /* IN */    *err_output,
                             dtype         /* OUT */    *err_input) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype vle = input[idx];
  err_input[idx] = err_output[idx] * rsqrt(vle * vle + 1);
}


#define tanhlog_d 3
#define tanhlog_a ((dtype)0.242528761112)
#define tanhlog_b ((dtype)305.459953195)

extern "C"
__global__ void forward_tanhlog(const dtype    /* IN */    *input,
                                dtype         /* OUT */    *output) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
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

extern "C"
__global__ void backward_tanhlog(const dtype    /* IN */    *input,
                                 const dtype    /* IN */    *output,
                                 const dtype    /* IN */    *err_output,
                                 dtype         /* OUT */    *err_input) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype x = input[idx];
  dtype xx = fabs(x);
  dtype y;
  if (xx <= tanhlog_d) {
    y = output[idx];
    y = y * y * (dtype)(-0.388484177) + (dtype)1.14381894;
  }
  else {
    y = fabs(tanhlog_a / x);
  }
  err_input[idx] = err_output[idx] * y;
}


extern "C"
__global__ void forward_sincos(const dtype    /* IN */    *input,
                               dtype         /* OUT */    *output) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype vle = input[idx];
  output[idx] = (idx & 1) ? sin(vle) : cos(vle);
}


extern "C"
__global__ void backward_sincos(const dtype    /* IN */    *input,
                                const dtype    /* IN */    *output,
                                const dtype    /* IN */    *err_output,
                                dtype         /* OUT */    *err_input) {
  size_t idx = blockIdx.x;  // we are running (N, 1, 1) (1, 1, 1) grid
  dtype vle = input[idx];
  err_input[idx] = err_output[idx] * ((idx & 1) ? cos(vle) : -sin(vle));
}
