// hack for highlighting syntax in OpenCL *.cl files without errors
#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#endif  // __OPENCL_VERSION__

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief forward propagation based on matrix multiplication.
 *
 * 1) C = A * B
 * 2) C_ij = bias_j, i = 0..(BATCH - 1), j = 0..(Y - 1)
 *
 * Should be defined externally:
 * <ul>
 * <li> dtype - data element type
 * <li> BLOCK_SIZE - size of the block for matrix multiplication,
 * <li> BATCH - minibatch size,
 * <li> H - input size,
 * <li> Y - output size.
 * <li> WEIGHTS_TRANSPOSED - if weights array is transposed (if necessary)
 * <li> ACTIVATION_<type> - activation type
 * <li> DOT - optimization flag for platforms with fast dot product build-in
 *            function
 * </ul>
 *
 * @param inputs layer inputs
 * @param weights layer weights
 * @param outputs layer outputs
 * @param bias vector to add to each line of output matrix
 */
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global const dtype /* IN*/ *inputs,
                __global const dtype /* IN*/ *weights,
                __global       dtype /*OUT*/ *outputs,
                __global const dtype /* IN*/ *bias) {
  dtype res = 0;
  int idx = 0;

  #ifndef WEIGHTS_TRANSPOSED
  #define B_COLS_DATA_PACK
  #endif  // WEIGHTS_TRANSPOSED

  #define A_ROWS BATCH
  #define A_COLS H
  #define B_ROWS H
  #define B_COLS Y
  #define A inputs
  #define B weights

  #include "matrix_mul.cl"

  #undef A_ROWS
  #undef A_COLS
  #undef B_ROWS
  #undef B_COLS
  #undef A
  #undef B

  if (is_in_range) {  // is_in_range is defined in matrix_mul.cl
    res += bias[get_global_id(0)];
    #ifdef ACTIVATION_LINEAR
      outputs[idx] = res;
    #elif ACTIVATION_TANH
      outputs[idx] = tanh(res * 0.6666) * 1.7159;
    #elif ACTIVATION_RELU
      outputs[idx] = res > 15 ? res : log(exp(res) + 1);
    #else
      #error "unsupported activation type"
    #endif  // ACTIVATION
  }
}
