#if INCLUDE_BIAS > 0
  if (!ty) {  // read from memory only for the first row
    AS[tx] = bias[bx * BLOCK_SIZE + tx];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum += AS[tx];
#endif

#if ACTIVATION_LINEAR > 0
  output[idx] = sum;
#elif ACTIVATION_TANH > 0
  output[idx] = tanh(sum * (dtype)0.6666) * (dtype)1.7159;
#elif ACTIVATION_RELU > 0
  output[idx] = sum > 15 ? sum : log(exp(sum) + 1);
#elif ACTIVATION_STRICT_RELU > 0
  output[idx] = max(sum, (dtype)0.0);
#elif ACTIVATION_SIGMOID > 0
  output[idx] = (dtype)1.0 / ((dtype)1.0 + exp(-sum));
#else
  #error "Activation function should be defined"
#endif
