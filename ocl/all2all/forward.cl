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
/// @param input input.
/// @param weights weights.
/// @param output output.
/// @param bias bias.
/// @details y = f(h * weights + bias)
///          Should be defined externally:
///          BLOCK_SIZE - for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global const dtype    /* IN */    *input,
                __global const dtype    /* IN */    *weights,
                #if INCLUDE_BIAS > 0
                __global const dtype    /* IN */    *bias,
                #endif
                __global dtype         /* OUT */    *output) {
  #define A_WIDTH BATCH
  #define B_WIDTH Y
  #define AB_COMMON H

  #define A input
  #define B weights

  #if WEIGHTS_TRANSPOSED > 0
  #define B_COL
  #endif

  #define STORE_OUTPUT "all2all/forward.store_output.cl"
  #include "matrix_multiplication.cl"

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B
}
