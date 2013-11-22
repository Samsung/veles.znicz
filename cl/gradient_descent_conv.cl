/* @brief Kernels for convolutional layer gradient descent.
 * @author Kazantsev Alexey <a.kazantsev@samsung.com>
 * @details Should be defined externally:
 *          BLOCK_SIZE - size of the block for matrix multiplication,
 *          BATCH - minibatch size,
 *          SX - input image width,
 *          SY - input image height,
 *          N_CHANNELS - number of input channels,
 *          KX - kernel width,
 *          KY - kernel height,
 *          N_KERNELS - number of kernels (i.e. neurons),
 *          REDUCE_SIZE - buffer size for reduce operation.
 */


#define ELEMENTS_PER_BLOCK (N_CHANNELS * KX * KY)
#define BLOCK_NUMBER (in_offs / ELEMENTS_PER_BLOCK)
#define OFFS_IN_BLOCK (in_offs % ELEMENTS_PER_BLOCK)
#define ROW_IN_BLOCK (OFFS_IN_BLOCK / (N_CHANNELS * KX))
#define COL_IN_BLOCK (OFFS_IN_BLOCK % (N_CHANNELS * KX))
#define BLOCKS_PER_SAMPLE ((SX - KX + 1) * (SY - KY + 1))
#define SAMPLE_NUMBER (BLOCK_NUMBER / BLOCKS_PER_SAMPLE)
#define BLOCK_IN_SAMPLE (BLOCK_NUMBER % BLOCKS_PER_SAMPLE)
#define ROW_IN_SAMPLE (BLOCK_IN_SAMPLE / (SX - KX + 1))
#define COL_IN_SAMPLE ((BLOCK_IN_SAMPLE % (SX - KX + 1)) * N_CHANNELS)
#define IN_REAL_OFFS ((SAMPLE_NUMBER * SY + ROW_IN_SAMPLE + ROW_IN_BLOCK) * (N_CHANNELS * SX) + (COL_IN_SAMPLE + COL_IN_BLOCK))


/// @brief Sets all elements of err_h to zero.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
__kernel
void err_h_clear(__global c_dtype /*OUT*/ *err_h) {
  err_h[get_global_id(0)] = c_from_re(0);
}


/// @brief Computes backprogated error for previous layer.
/// @param err_y backpropagated error of the output layer.
/// @param weights weights.
/// @param err_h resulted backpropagated error for previous layer.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @details err_h_tmp = err_y * weights
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IN*/ *weights,
                  __global c_dtype /*OUT*/ *err_h) {

  #define A_WIDTH (BATCH * (SX - KX + 1) * (SY - KY + 1))
  #define B_WIDTH ELEMENTS_PER_BLOCK
  #define AB_COMMON N_KERNELS

  #define A err_y
  #define B weights
  #define C err_h_tmp

  #ifndef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  MX_MUL

  #ifndef WEIGHTS_TRANSPOSED
  #undef B_COL
  #endif

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B
  #undef C

  if (valid) {
    #define in_offs idx
    ATOM_ADD(&err_h[IN_REAL_OFFS], sum[0]);
  }
}


/// @brief Calculate gradient for weights update.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param err_y backpropagated error
/// @param h layer input
/// @param weights layer weights
/// @param gradient computed gradient to store in if not null
/// @param alpha_batch (-global_alpha / batch_size)
/// @param alpha_lambda (-global_alpha * global_lambda)
/// @details gradient = err_y * h * alpha_batch + weights * alpha_lambda
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IN*/  *h,
                    __global c_dtype /*IO*/ *weights, __global c_dtype /*OUT*/ *gradient,
                    const dtype alpha_batch, const dtype alpha_lambda) {
  #ifdef WEIGHTS_TRANSPOSED

  #define A_WIDTH ELEMENTS_PER_BLOCK
  #define B_WIDTH N_KERNELS
  #define A h
  #define B err_y

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS

  #else

  #define A_WIDTH N_KERNELS
  #define B_WIDTH ELEMENTS_PER_BLOCK
  #define A err_y
  #define B h

  #define in_offs b_offs
  #define B_REAL_OFFS IN_REAL_OFFS

  #endif

  #define AB_COMMON (BATCH * (SX - KX + 1) * (SY - KY + 1))
  #define C weights

  #define A_COL
  #define B_COL

  MX_MUL

  #undef A_COL
  #undef B_COL

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B
  #undef C

  if (valid) {
    c_dtype weight = weights[idx];
    c_dtype gd = sum[0] * alpha_batch + weight * alpha_lambda;
    #ifdef STORE_GRADIENT
    gradient[idx] = gd;
    #endif
    #ifdef APPLY_GRADIENT
    weights[idx] = weight + gd;
    #endif
  }
}


/// @brief Calculate gradient for bias update.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param bias layer bias
/// @param err_y backpropagated error
/// @param gradient computed gradient to store in if not null
/// @param alpha_batch (-global_alpha / batch_size)
/// @details gradient = sum(err_y) * alpha_batch
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void bias_update(__global c_dtype /*IN*/ *err_y, __global c_dtype /*IO*/ *bias,
                 __global c_dtype /*OUT*/ *gradient, const dtype alpha_batch) {

  #define A err_y
  #define A_WIDTH N_KERNELS
  #define A_HEIGHT (BATCH * (SX - KX + 1) * (SY - KY + 1))
  #define A_COL

  MX_REDUCE

  #undef A_COL
  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  if (!tx) {
    sum += AS[0];

    c_dtype gd = sum * alpha_batch;
    #ifdef STORE_GRADIENT
    gradient[bx] = gd;
    #endif
    #ifdef APPLY_GRADIENT
    bias[bx] += gd;
    #endif
  }
}
