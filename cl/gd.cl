//TODO(a.kazantsev): optimize for sum precision like in feed.cl

//Should be defined externally:
//#define BLOCK_SIZE 1
//#define BATCH 178
//#define H 13
//#define Y 5

#define B_WIDTH H
#define AB_COMMON Y
/*
	C = A * B
	
	(row * column) here
	
	We are calculating values for block of matrix C for each workgroup.
	
	Example:
	[178][5] * [5][13] = [178][13]
	size_t WorkSize[2] = {13, 178}
	size_t LocalSize[2] = {BLOCK_SIZE, BLOCK_SIZE}
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void err_h_update(__global float *A, __global float *B, __global float *C)
{
 __local float AS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of A
 __local float BS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of B
 
 // Block index in matrix C, where the values will be put
 int bx = get_group_id(0); // from 0 to B_WIDTH / BLOCK_SIZE - 1
 int by = get_group_id(1); // from 0 to A_HEIGHT / BLOCK_SIZE - 1
 
 // Thread index, each thread calculates one element of the resulted submatrix
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 float sum = 0.0f;
 
 int offs_in_the_block = ty * AB_COMMON/*row in the block*/ + tx/*column in the block*/;
 int a_offs = by * AB_COMMON * BLOCK_SIZE/*row start*/ + offs_in_the_block;
 int b_offs = ty * B_WIDTH + bx * BLOCK_SIZE + tx;
 
 for(int i = 0; i < AB_COMMON / BLOCK_SIZE; i++, a_offs += BLOCK_SIZE, b_offs += B_WIDTH * BLOCK_SIZE)
 {
  AS[ty][tx] = A[a_offs];
  BS[ty][tx] = B[b_offs];
  
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);
  
  #pragma unroll
  for(int k = 0; k < BLOCK_SIZE; k++)
   sum += AS[ty][k] * BS[k][tx];
  
  // ensure we can reload shared with new values
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 
 C[get_global_id(1) * B_WIDTH + get_global_id(0)] = sum;
}
#undef B_WIDTH
#undef AB_COMMON

#define B_WIDTH H
#define AB_COMMON BATCH
#define A_WIDTH Y
/*
	W = W * r + A * B * k
	
	We are calculating values for block of matrix W for each workgroup.
	
	(column * column) here
	
	A - err_y (error on output layer)
	B - h (input layer values)
	W - weights
	k_ = (-global_alpha / batch_size)
	r_ = 1.0 + (-global_alpha * global_lambda)
	
	Example:
	[60000][16] * [60000][32] = [32][16]
	size_t WorkSize[2] = {32, 16}
	size_t LocalSize[2] = {BLOCK_SIZE, BLOCK_SIZE}
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global float *A, __global float *B, __global float *W, const float k_, const float r_)
{
 __local float AS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of A
 __local float BS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of B
 
 // Block index in matrix C, where the values will be put
 int bx = get_group_id(0); // from 0 to B_WIDTH / BLOCK_SIZE - 1
 int by = get_group_id(1); // from 0 to A_WIDTH / BLOCK_SIZE - 1
 
 // Thread index, each thread calculates one element of the resulted submatrix
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 float sum = 0.0f;
 
 int a_offs = ty * A_WIDTH + by * BLOCK_SIZE + tx;
 int b_offs = ty * B_WIDTH + bx * BLOCK_SIZE + tx;
 
 for(int i = 0; i < AB_COMMON / BLOCK_SIZE; i++, a_offs += A_WIDTH * BLOCK_SIZE, b_offs += B_WIDTH * BLOCK_SIZE)
 {
  AS[ty][tx] = A[a_offs];
  BS[ty][tx] = B[b_offs];
  
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);
  
  #pragma unroll
  for(int k = 0; k < BLOCK_SIZE; k++)
   sum += AS[k][ty] * BS[k][tx];
  
  // ensure we can reload shared with new values
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 
 int idx = get_global_id(1) * B_WIDTH + get_global_id(0);
 W[idx] = W[idx] * r_ + sum * k_;
}
#undef B_WIDTH
#undef AB_COMMON
#undef A_WIDTH

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
