//Should be defined externally:
//#define BLOCK_SIZE 16
//#define BATCH 178
//#define H 13
//#define Y 5


/*
	err_h = err_y * weights
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global float *err_y, __global float *weights, __global float *err_h)
{
 #define A_WIDTH BATCH
 #define B_WIDTH H
 #define AB_COMMON Y

 #define A err_y
 #define B weights
 #define C err_h

 #define B_COL

 MX_MUL

 #undef B_COL

 #undef A_WIDTH
 #undef B_WIDTH
 #undef AB_COMMON

 #undef A
 #undef B
 #undef C

 err_h[idx] = sum[0];
}


/*
	weights = weights * r_ + err_y * h * k_

	k_ = (-global_alpha / batch_size)
	r_ = 1.0 + (-global_alpha * global_lambda)
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global float *err_y, __global float *h, __global float *weights,
                    const float k_, const float r_)
{
 #define A_WIDTH Y
 #define B_WIDTH H
 #define AB_COMMON BATCH

 #define A err_y
 #define B h
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

 weights[idx] = weights[idx] * r_ + sum[0] * k_;
}


/*
	bias += err_y * k
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void bias_update(__global float *bias, __global float *err_y, const float k_)
{
 __local float AS[BLOCK_SIZE][BLOCK_SIZE];
 
 int bx = get_group_id(0); // from 0 to Y / BLOCK_SIZE - 1
 
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 float sum = 0.0f;
 
 int offs = get_global_id(0) + ty * Y;
 for(int i = 0; i < BATCH / BLOCK_SIZE; i++, offs += Y * BLOCK_SIZE)
 {
  sum += err_y[offs];
 }
 
 AS[ty][tx] = sum;
 // ensure all shared loaded
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(!ty)
 {
  sum = AS[0][tx];
  
  #pragma unroll
  for(int k = 1; k < BLOCK_SIZE; k++)
   sum += AS[k][tx];
  
  bias[get_global_id(0)] += sum * k_;
 }
}
