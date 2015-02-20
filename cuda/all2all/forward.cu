#ifndef OUTPUT_SIZE
#error "OUTPUT_SIZE must be defined"
#endif

#ifndef BIAS_SIZE
#error "BIAS_SIZE must be defined"
#endif

extern "C"
__global__ void apply_bias_with_activation(dtype *output, const dtype *bias) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < OUTPUT_SIZE) {
    dtype y = output[idx];
    #if INCLUDE_BIAS > 0
      y += bias[idx % BIAS_SIZE];
    #endif
    #if ACTIVATION_LINEAR > 0
      output[idx] = y;
    #elif ACTIVATION_TANH > 0
      output[idx] = (dtype)1.7159 * tanh((dtype)0.6666 * y);
    #elif ACTIVATION_RELU > 0
      output[idx] = y > 15 ? y : log(exp(y) + 1);
    #elif ACTIVATION_STRICT_RELU > 0
      output[idx] = max(y, (dtype)0.0);
    #elif ACTIVATION_SIGMOID > 0
      output[idx] = (dtype)1.0 / ((dtype)1.0 + exp(-y));
    #else
      #error "Unsupported activation"
    #endif
  }
}
