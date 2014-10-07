#include "gradient_descent_common.cl"


/// @brief Computes backprogated error for previous layer:
///        err_input = err_output * weights.
/// @details Should be defined externally:
///          A_BLOCK_SIZE, B_BLOCK_SIZE, COMMON_BLOCK_SIZE - for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(B_BLOCK_SIZE, A_BLOCK_SIZE, 1)))
void err_input_update(__global const dtype    /* IN */    *err_output,
                      __global const dtype    /* IN */    *weights,
                      __global dtype         /* OUT */    *err_input) {
  #define A_WIDTH BATCH
  #define B_WIDTH H
  #define AB_COMMON Y

  #define A err_output
  #define B weights

  #if WEIGHTS_TRANSPOSED <= 0
  #define B_COL
  #endif

  #include "matrix_multiplication.cl"

  #if WEIGHTS_TRANSPOSED <= 0
  #undef B_COL
  #endif

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  if (valid) {
    err_input[idx] = sum;
  }
}
