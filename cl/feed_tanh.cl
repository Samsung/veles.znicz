/*
	Feeds the layer with scaled tanh() activation f(): y = 1.7159 * tanh(0.6666 * (W * x + b))
		Because: f(1) = 1, f(-1) = -1, f"(x) maximum at x = 1
	
	C = f(A * B + bias_weights)
	
	One of matrices is transposed, so we should perform row x row multiplication.
	We are calculating values for block of matrix C for each workgroup.
	
	Example:
	[16][1024] * [2048][1024] = [16][2048]
	size_t WorkSize[2] = {2048, 16}
	size_t LocalSize[2] = {BLOCK_SIZE, BLOCK_SIZE}
*/
//TODO(a.kazantsev): optimize further.
#ifndef N_SUM
#if AB_WIDTH <= 32
#define N_SUM 2
#endif
#if (AB_WIDTH > 32) && (AB_WIDTH <= 64)
#define N_SUM 4
#endif
#if (AB_WIDTH > 64) && (AB_WIDTH <= 128)
#define N_SUM 8
#endif
#if (AB_WIDTH > 128) && (AB_WIDTH <= 256)
#define N_SUM 16
#endif
#if (AB_WIDTH > 256) && (AB_WIDTH <= 512)
#define N_SUM 32
#endif
#if (AB_WIDTH > 512) && (AB_WIDTH <= 1024)
#define N_SUM 64
#endif
#if AB_WIDTH > 1024
#define N_SUM 128
#endif
#endif
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void FEED_LAYER(__global float *A, __global float *B, __global float *C, __global float *bias_weights)
{
 __local float AS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of A
 __local float BS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of B
 
 // Block index in matrix C, where the values will be put
 int bx = get_group_id(0); // from 0 to B_HEIGHT / BLOCK_SIZE - 1
 int by = get_group_id(1); // from 0 to A_HEIGHT / BLOCK_SIZE - 1
 
 // Thread index, each thread calculates one element of the resulted submatrix
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 int offs_in_the_block = ty * AB_WIDTH/*row in the block*/ + tx/*column in the block*/;
 int a_offs = by * AB_WIDTH * BLOCK_SIZE/*row start*/ + offs_in_the_block;
 int b_offs = bx * AB_WIDTH * BLOCK_SIZE/*row start*/ + offs_in_the_block;
 
 float sum[N_SUM];
 for(int i_sum = 0; i_sum < N_SUM; i_sum++)
 {
  sum[i_sum] = 0.0f;
  for(int i = AB_WIDTH / BLOCK_SIZE * i_sum / N_SUM; i < AB_WIDTH / BLOCK_SIZE * (i_sum + 1) / N_SUM; i++)
  {
   AS[ty][tx] = A[i * BLOCK_SIZE/*block offset in the row*/ + a_offs];
   BS[ty][tx] = B[i * BLOCK_SIZE/*block offset in the row*/ + b_offs];
   
   // ensure all shared loaded
   barrier(CLK_LOCAL_MEM_FENCE);
   
   #pragma unroll
   for(int k = 0; k < BLOCK_SIZE; k++)
    sum[i_sum] += AS[ty][k] * BS[tx][k];
   
   // ensure we can reload shared with new values
   barrier(CLK_LOCAL_MEM_FENCE);
  }
 }
 for(int n_sum = N_SUM; n_sum > 2; n_sum >>= 1)
 {
  for(int i_sum = 0; i_sum < (n_sum >> 1); i_sum++)
   sum[i_sum] = sum[i_sum << 1] + sum[(i_sum << 1) + 1];
 }
 sum[0] += sum[1];
 
 if(!ty) // read from memory only for the first row
  AS[0][tx] = bias_weights[bx * BLOCK_SIZE + tx];
 
 barrier(CLK_LOCAL_MEM_FENCE);
 
 C[get_global_id(1) * B_HEIGHT + get_global_id(0)
   /* same as:
		by * B_HEIGHT * BLOCK_SIZE + bx * BLOCK_SIZE + ty * B_HEIGHT + tx
   */] = 1.7159f * tanh(0.6666f * (sum[0] + AS[0][tx]));
}
