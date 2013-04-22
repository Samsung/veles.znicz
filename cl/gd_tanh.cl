/*
	C = A * B
	
	We are calculating values for block of matrix C for each workgroup.
	
	Example:
	[60000][16] * [16][32] = [60000][32]
	size_t WorkSize[2] = {32, 60000}
	size_t LocalSize[2] = {BLOCK_SIZE, BLOCK_SIZE}
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void mx_mul(__global float *A, __global float *B, __global float *C)
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
