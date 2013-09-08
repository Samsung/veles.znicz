/*
 * Gradient descent.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


//Should be declared externally:
//#define dtype float
//#define BLOCK_SIZE 16
//#define BATCH 178
//#define H 13
//#define Y 5


/// @brief err_h = err_y * weights
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global dtype *err_y, __global dtype *weights, __global dtype *err_h) {
  #define A_WIDTH BATCH
  #define B_WIDTH H
  #define AB_COMMON Y

  #define A err_y
  #define B weights
  #define C err_h

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

  err_h[idx] = sum[0];
}


/// @brief Calculate gradient for weights update.
/// @param err_y backpropagated error
/// @param h layer input
/// @param weights layer weights
/// @param gradient computed gradient to store in if not null
/// @param alpha_batch (-global_alpha / batch_size)
/// @param alpha_lambda (-global_alpha * global_lambda)
/// @details gradient = err_y * h * alpha_batch + weights * alpha_lambda
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global dtype /*IN*/ *err_y, __global dtype /*IN*/  *h,
                    __global dtype /*IO*/ *weights, __global dtype /*OUT*/ *gradient,
                    const dtype alpha_batch, const dtype alpha_lambda) {
  #ifdef WEIGHTS_TRANSPOSED
  #define A_WIDTH H
  #define B_WIDTH Y
  #define A h
  #define B err_y
  #else
  #define A_WIDTH Y
  #define B_WIDTH H
  #define A err_y
  #define B h
  #endif

  #define AB_COMMON BATCH
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

  dtype weight = weights[idx];
  dtype gd = sum[0] * alpha_batch + weight * alpha_lambda;
  #ifdef STORE_GRADIENT
  gradient[idx] = gd;
  #endif
  #ifdef APPLY_GRADIENT
  weights[idx] = weight + gd;
  #endif
}


/// @brief Calculate gradient for bias update.
/// @param bias layer bias
/// @param err_y backpropagated error
/// @param gradient computed gradient to store in if not null
/// @param alpha_batch (-global_alpha / batch_size)
/// @details gradient = sum(err_y) * alpha_batch
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void bias_update(__global dtype /*IN*/ *err_y, __global dtype /*IO*/ *bias,
                 __global dtype /*OUT*/ *gradient, const dtype alpha_batch) {
  __local dtype AS[BLOCK_SIZE][BLOCK_SIZE];
 
  int bx = get_group_id(0); // from 0 to Y / BLOCK_SIZE - 1
 
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
  int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
  dtype sum = 0.0f;
 
  int offs = get_global_id(0) + ty * Y;
  for (int i = 0; i < BATCH / BLOCK_SIZE; i++, offs += Y * BLOCK_SIZE) {
    sum += err_y[offs];
  }
 
  AS[ty][tx] = sum;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);
 
  if (!ty) {
    sum = AS[0][tx];
  
    #pragma unroll
    for (int k = 1; k < BLOCK_SIZE; k++)
      sum += AS[k][tx];

    dtype gd = sum * alpha_batch;
    #ifdef STORE_GRADIENT
    gradient[get_global_id(0)] = gd;
    #endif
    #ifdef APPLY_GRADIENT
    bias[get_global_id(0)] += gd;
    #endif
  }
}
