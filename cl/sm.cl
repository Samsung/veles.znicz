//Should be defined externally:
//#define BLOCK_SIZE 24
//#define BATCH 192
//#define H 24
//#define Y 24
//#define Y_REAL 5


/*
	For each sample from batch of y:
	1. Find m = max().
	2. Overwrite x as exp(x - m).
	3. Compute sum of all x.
	4. Divide x by sum.
	
	Example:
		y = [178][3].
		178 - batch size.
		
		if BLOCK_SIZE = 24:
			global_size = [24, 192]
			local_size = [24, 24]
*/
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
//TODO(a.kazantsev): add #if for the case when Y <= BLOCK_SIZE (which usually is)
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void apply_exp(__global float *y)
{
 __local float AS[BLOCK_SIZE][BLOCK_SIZE];
 __local float MS[BLOCK_SIZE], SUMS[BLOCK_SIZE];
 
 int by = get_group_id(1); // from 0 to BATCH / BLOCK_SIZE - 1
 
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 int start_offs = (by * BLOCK_SIZE + ty) * Y + tx;
 
 float m = -1.0e30f;
 int offs = start_offs;
 for(int i = 0; i < Y_REAL / BLOCK_SIZE; i++, offs += BLOCK_SIZE)
 {
  m = max(m, y[offs]);
 }
 if(tx < Y_REAL % BLOCK_SIZE)
  m = max(m, y[offs]);
 AS[ty][tx] = m;
 // ensure all shared loaded
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(!tx)
 {
  m = AS[ty][0];
  
  #pragma unroll
  for(int k = 1; k < MIN(BLOCK_SIZE, Y_REAL); k++)
   m = max(m, AS[ty][k]);
  
  MS[ty] = m;
 }
 // ensure max computed
 barrier(CLK_LOCAL_MEM_FENCE);
 m = MS[ty];
 
 float sum = 0.0f;
 offs = start_offs;
 for(int i = 0; i < Y_REAL / BLOCK_SIZE; i++, offs += BLOCK_SIZE)
 {
  sum += exp(y[offs] - m);
 }
 if(tx < Y_REAL % BLOCK_SIZE)
  sum += exp(y[offs] - m);
 AS[ty][tx] = sum;
 // ensure all shared loaded
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(!tx)
 {
  sum = AS[ty][0];
  
  #pragma unroll
  for(int k = 1; k < MIN(BLOCK_SIZE, Y_REAL); k++)
   sum += AS[ty][k];
  
  SUMS[ty] = sum;
 }
 // ensure sum computed
 barrier(CLK_LOCAL_MEM_FENCE);
 sum = SUMS[ty];
 
 offs = start_offs;
 for(int i = 0; i < Y_REAL / BLOCK_SIZE; i++, offs += BLOCK_SIZE)
 {
  y[offs] = exp(y[offs] - m) / sum;
 }
 if(tx < Y_REAL % BLOCK_SIZE)
  y[offs] = exp(y[offs] - m) / sum;
}
