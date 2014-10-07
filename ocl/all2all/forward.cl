#include "defines.cl"
#include "highlight.cl"

#ifndef INCLUDE_BIAS
#error "INCLUDE_BIAS should be defined"
#endif

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

/// @brief Feeds all-to-all layer with activation function:
///        linear activation: x;
///        scaled tanh activation: 1.7159 * tanh(0.6666 * x),
///        because: f(1) = 1, f(-1) = -1 and f"(x) maximum at x = 1.
/// @param h input.
/// @param weights weights.
/// @param y output.
/// @param bias bias.
/// @details y = f(h * weights + bias)
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(B_BLOCK_SIZE, A_BLOCK_SIZE, 1)))
void feed_layer(__global const dtype    /* IN */    *h,
                __global const dtype    /* IN */    *weights,
                #if INCLUDE_BIAS > 0
                __global const dtype    /* IN */    *bias,
                #endif
                __global dtype         /* OUT */    *y) {
  #define A_WIDTH BATCH
  #define B_WIDTH Y
  #define AB_COMMON H

  #define A h
  #define B weights

  #if WEIGHTS_TRANSPOSED > 0
  #define B_COL
  #endif

  #include "matrix_multiplication.cl"

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  #if INCLUDE_BIAS > 0
  if ((valid) && (!ty)) {  // read from memory only for the first row
    AS[tx] = bias[bx * B_BLOCK_SIZE + tx];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum += AS[tx];
  #endif

  if (valid) {
 	  #if ACTIVATION_LINEAR > 0
      y[idx] = sum;
 	  #elif ACTIVATION_TANH > 0
      y[idx] = tanh(sum * (dtype)0.6666) * (dtype)1.7159;
 	  #elif ACTIVATION_RELU > 0
      y[idx] = sum > 15 ? sum : log(exp(sum) + 1);
    #elif ACTIVATION_STRICT_RELU > 0
      y[idx] = max(sum, (dtype)0.0);
    #else
      #error "Activation function should be defined"
    #endif
  }
}
