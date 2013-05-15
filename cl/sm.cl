/*
	For each sample from batch of y:
	1. Find m = max().
	2. Overwrite x as exp(x - m).
	3. Compute sum of all x.
	4. Divide x by sum.
	
	Example:
		y = [178][3].
		178 - batch size.
*/
#define Y B_HEIGHT
#define Y_REAL B_HEIGHT_REAL
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
//TODO(a.kazantsev): add #if for the case when Y <= BLOCK_SIZE (which usually is)
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void apply_exp(__global float *y)
{
 __local float AS[BLOCK_SIZE][BLOCK_SIZE];
 
 int bx = get_group_id(0); // from 0 to BATCH / BLOCK_SIZE - 1
 
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 int start_offs = (bx * BLOCK_SIZE + ty) * Y + tx;
 
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
  
  AS[ty][0] = m;
 }
 // ensure max computed
 barrier(CLK_LOCAL_MEM_FENCE);
 m = AS[ty][0];
 
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
  
  AS[ty][0] = sum;
 }
 // ensure sum computed
 barrier(CLK_LOCAL_MEM_FENCE);
 sum = AS[ty][0];
 
 offs = start_offs;
 for(int i = 0; i < Y_REAL / BLOCK_SIZE; i++, offs += BLOCK_SIZE)
 {
  y[offs] = exp(y[offs] - m) / sum;
 }
 if(tx < Y_REAL % BLOCK_SIZE)
  y[offs] = exp(y[offs] - m) / sum;
}
#undef Y
#undef Y_REAL
