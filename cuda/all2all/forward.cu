#ifndef OUTPUT_SAMPLE_SIZE
#error "OUTPUT_SAMPLE_SIZE must be defined"
#endif

extern "C"
__global__ void apply_bias_with_activation(dtype *output, const dtype *bias) {
  size_t idx = blockIdx.y * OUTPUT_SAMPLE_SIZE + blockIdx.x;
  dtype y = output[idx];
  #if INCLUDE_BIAS > 0
    y += bias[blockIdx.x];
  #endif
  #if ACTIVATION_LINEAR > 0
    output[idx] = y;
  #elif ACTIVATION_TANH > 0
    output[idx] = (dtype)1.7159 * tanh((dtype)0.6666 * y);
  #elif ACTIVATION_RELU > 0
    output[idx] = y > 15 ? y : log(exp(y) + 1);
  #elif ACTIVATION_STRICT_RELU > 0
    output[idx] = max(y, (dtype)0.0);
  #else
    #error "Unsupported activation"
  #endif
}
