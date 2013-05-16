/*
	Define for matrix multiplication.

	Example of how to use:
	1. Read this file to variable s_mx_mul.
	2. Read your other source files to variable s.
	3. Replace all occurencies of MX_MUL within s with s_mx_mul.

	Kernel should be defined as:
	__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))

	Sizes should be defined externally (values are given for example):
	#define BLOCK_SIZE 16
	#define A_WIDTH 512
	#define B_WIDTH 256
	#define AB_COMMON 131072

	As well as Matricies:
	#define A err_y
	#define B h
	#define C weights

	And column order if neccessary (otherwise row order is assumed):
	#define A_COL
	#define B_COL

	C = A * B

	We will calculate values for block of matrix C for each workgroup.

	[AB_COMMON][A_WIDTH] * [B_WIDTH][AB_COMMON] = [A_WIDTH][B_WIDTH]
	size_t WorkSize[2] = {B_WIDTH, A_WIDTH}
	size_t LocalSize[2] = {BLOCK_SIZE, BLOCK_SIZE}
*/
#ifndef N_SUM
#define BLOCKS (AB_COMMON / BLOCK_SIZE)
#if BLOCKS <= 4
#define N_SUM 2
#endif
#if (BLOCKS > 4) && (BLOCKS <= 8)
#define N_SUM 4
#endif
#if (BLOCKS > 8) && (BLOCKS <= 16)
#define N_SUM 8
#endif
#if (BLOCKS > 16) && (BLOCKS <= 32)
#define N_SUM 16
#endif
#if (BLOCKS > 32) && (BLOCKS <= 64)
#define N_SUM 32
#endif
#if (BLOCKS > 64) && (BLOCKS <= 128)
#define N_SUM 64
#endif
#if BLOCKS > 128
#define N_SUM 128
#endif
#endif

// The source for matrix multiplication comes here:
 __local float AS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of A
 __local float BS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of B
 
 // Block index in matrix C, where the values will be put
 int bx = get_group_id(0); // from 0 to B_WIDTH / BLOCK_SIZE - 1
 int by = get_group_id(1); // from 0 to A_WIDTH / BLOCK_SIZE - 1
 
 // Thread index, each thread calculates one element of the resulted submatrix
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
#ifdef A_COL
 int a_offs = ty * A_WIDTH + by * BLOCK_SIZE + tx;
#define A_OFFS A_WIDTH * BLOCK_SIZE
#else
 int a_offs = by * AB_COMMON * BLOCK_SIZE + ty * AB_COMMON + tx;
#define A_OFFS BLOCK_SIZE
#endif

#ifdef B_COL
#define B_OFFS B_WIDTH * BLOCK_SIZE
 int b_offs = ty * B_WIDTH + bx * BLOCK_SIZE + tx;
#else
#define B_OFFS BLOCK_SIZE
 int b_offs = bx * AB_COMMON * BLOCK_SIZE + ty * AB_COMMON + tx;
#endif

 float sum[N_SUM];
 for(int i_sum = 0; i_sum < N_SUM; i_sum++)
 {
  sum[i_sum] = 0.0f;
  for(int i = AB_COMMON / BLOCK_SIZE * i_sum / N_SUM; i < AB_COMMON / BLOCK_SIZE * (i_sum + 1) / N_SUM; i++,
      a_offs += A_OFFS, b_offs += B_OFFS)
  {
   AS[ty][tx] = A[a_offs];
   BS[ty][tx] = B[b_offs];

   // ensure all shared loaded
   barrier(CLK_LOCAL_MEM_FENCE);

   #pragma unroll
   for(int k = 0; k < BLOCK_SIZE; k++)
#ifdef B_COL
#ifdef A_COL
    sum[i_sum] += AS[k][ty] * BS[k][tx];
#else
    sum[i_sum] += AS[ty][k] * BS[k][tx];
#endif
#else
#ifdef A_COL
    sum[i_sum] += AS[k][ty] * BS[tx][k];
#else
    sum[i_sum] += AS[ty][k] * BS[tx][k];
#endif
#endif

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

 int idx = get_global_id(1) * B_WIDTH + get_global_id(0)
   /* same as:
		by * B_HEIGHT * BLOCK_SIZE + bx * BLOCK_SIZE + ty * B_HEIGHT + tx
   */;

#undef A_OFFS
#undef B_OFFS
// The source for matrix multiplication ends here (the result will be in sum[0]).
