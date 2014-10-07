#include "gradient_descent_common.cl"
#include "conv_common.cl"


#ifdef USE_ATOMICS
/* @brief Kernels for convolutional layer gradient descent.
 * @details Should be defined externally:
 *          defines from conv_common.cl,
 *          REDUCE_SIZE - buffer size for reduce operation.
 */

/// @brief Computes backprogated error for previous layer.
/// @param err_y backpropagated error of the output layer.
/// @param weights weights.
/// @param err_h resulted backpropagated error for previous layer.
/// @details err_h = err_y * weights.
__kernel __attribute__((reqd_work_group_size(B_BLOCK_SIZE, A_BLOCK_SIZE, 1)))
void err_input_update(__global const dtype    /* IN */    *err_output,
                      __global const dtype    /* IN */    *weights,
                      __global dtype         /* OUT */    *err_input) {

  #define A_WIDTH (BATCH * KERNELS_PER_SAMPLE)
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define AB_COMMON N_KERNELS

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

  #define in_offs idx
  if ((valid) && (IN_REAL_OFFS_VALID)) {
    ATOM_ADD(&err_input[IN_REAL_OFFS], sum);
  }
  #undef in_offs
}
#endif


KERNEL_CLEAR(err_input_clear, dtype)
