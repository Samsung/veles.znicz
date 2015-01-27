#include "defines.cu"


extern "C"
__global__ void forward_tanh(const dtype    *input,
                             dtype          *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    output[idx] = tanh(input[idx] * (dtype)0.6666) * (dtype)1.7159;
  }
}


extern "C"
__global__ void backward_tanh(const dtype    *input,
                              const dtype    *output,
                              const dtype    *err_output,
                              dtype          *err_input) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype vle = output[idx];
    err_input[idx] = err_output[idx] * (vle * vle * (dtype)(-0.388484177) + (dtype)1.14381894);
  }
}


extern "C"
__global__ void forward_mul(const dtype    *input,
                            dtype          *output,
                            const dtype    factor) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    output[idx] = input[idx] * factor;
  }
}


extern "C"
__global__ void backward_mul(const dtype    *input,
                             const dtype    *output,
                             const dtype    *err_output,
                             dtype          *err_input,
                             const dtype    factor) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    err_input[idx] = err_output[idx] * factor;
  }
}


extern "C"
__global__ void forward_relu(const dtype    *input,
                             dtype          *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype vle = input[idx];
    output[idx] = vle > 15 ? vle : log(exp(vle) + 1);
  }
}


extern "C"
__global__ void backward_relu(const dtype    *input,
                              const dtype    *output,
                              const dtype    *err_output,
                              dtype          *err_input) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    err_input[idx] = err_output[idx] * ((dtype)1.0 - exp(-output[idx]));
  }
}


extern "C"
__global__ void forward_strict_relu(const dtype    *input,
                                    dtype          *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    output[idx] = max(input[idx], (dtype)0);
  }
}


extern "C"
__global__ void backward_strict_relu(const dtype    *input,
                                     const dtype    *output,
                                     const dtype    *err_output,
                                     dtype          *err_input) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    err_input[idx] = output[idx] > 0 ? err_output[idx] : 0;
  }
}


extern "C"
__global__ void forward_log(const dtype    *input,
                            dtype          *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype vle = input[idx];
    output[idx] = log(vle + sqrt(vle * vle + 1));
  }
}


extern "C"
__global__ void backward_log(const dtype    *input,
                             const dtype    *output,
                             const dtype    *err_output,
                             dtype          *err_input) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype vle = input[idx];
    err_input[idx] = err_output[idx] * rsqrt(vle * vle + 1);
  }
}


#define tanhlog_d 3
#define tanhlog_a ((dtype)0.242528761112)
#define tanhlog_b ((dtype)305.459953195)

extern "C"
__global__ void forward_tanhlog(const dtype    *input,
                                dtype          *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
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
}

extern "C"
__global__ void backward_tanhlog(const dtype    *input,
                                 const dtype    *output,
                                 const dtype    *err_output,
                                 dtype          *err_input) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
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
}


extern "C"
__global__ void forward_sincos(const dtype    *input,
                               dtype          *output) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype vle = input[idx];
    output[idx] = (idx & 1) ? sin(vle) : cos(vle);
  }
}


extern "C"
__global__ void backward_sincos(const dtype    *input,
                                const dtype    *output,
                                const dtype    *err_output,
                                dtype          *err_input) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype vle = input[idx];
    err_input[idx] = err_output[idx] * ((idx & 1) ? cos(vle) : -sin(vle));
  }
}
