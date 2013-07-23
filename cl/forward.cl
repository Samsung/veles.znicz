/*
 * Forward propagation.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


//Should be declared externally:
//#define dtype float
//#define BLOCK_SIZE 16
//#define BATCH 178
//#define H 13
//#define Y 5


/// @brief Feeds the layer with linear or scaled tanh() activation f(): y = 1.7159 * tanh(0.6666 * (W * x + b))
///        Because: f(1) = 1, f(-1) = -1, f"(x) maximum at x = 1
/// @param h input.
/// @param weights weights.
/// @param y output.
/// @param bias bias.
/// @details y = f(h * weights + bias)
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void FEED_LAYER(__global dtype /*IN*/ *h, __global dtype /*IN*/ *weights,
                __global dtype /*OUT*/ *y, __global dtype /*IN*/ *bias) {
  #define A_WIDTH BATCH
  #define B_WIDTH Y
  #define AB_COMMON H

  #define A h
  #define B weights
  #define C y

  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  MX_MUL

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B
  #undef C

  if(!ty) // read from memory only for the first row
    AS[0][tx] = bias[bx * BLOCK_SIZE + tx];

  barrier(CLK_LOCAL_MEM_FENCE);

  y[idx] =
 	#ifdef ACTIVATION_LINEAR
    sum[0] + AS[0][tx];
 	#endif
 	#ifdef ACTIVATION_TANH
 		1.7159f * tanh(0.6666f * (sum[0] + AS[0][tx]));
 	#endif
}
