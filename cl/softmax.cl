/*
 * Applies softmax exponent.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


//Should be declared externally:
//#define dtype float
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
void apply_exp(__global dtype *y, __global itype *max_idx)
{
 __local dtype AS[BLOCK_SIZE][BLOCK_SIZE];
 __local itype IS[BLOCK_SIZE][BLOCK_SIZE];
 __local dtype MS[BLOCK_SIZE], SUMS[BLOCK_SIZE];
 
 int by = get_group_id(1); // from 0 to BATCH / BLOCK_SIZE - 1
 
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 int i_sample = by * BLOCK_SIZE + ty;
 int sample_offs = i_sample * Y;
 int start_offs = sample_offs + tx;
 
 dtype m = -1.0e30f;
 int im = 0;
 int offs = start_offs;
 for(int i = 0; i < Y_REAL / BLOCK_SIZE; i++, offs += BLOCK_SIZE)
 {
  //m = max(m, y[offs]);
  dtype vle = y[offs];
  if(m < vle)
  {
   m = vle;
   im = offs - sample_offs;
  }
 }
 if(tx < Y_REAL % BLOCK_SIZE)
 {
  //m = max(m, y[offs]);
  dtype vle = y[offs];
  if(m < vle)
  {
   m = vle;
   im = offs - sample_offs;
  }
 }
 AS[ty][tx] = m;
 IS[ty][tx] = im;
 // ensure all shared loaded
 barrier(CLK_LOCAL_MEM_FENCE);
 
 if(!tx)
 {
  m = AS[ty][0];
  im = IS[ty][0];
  
  #pragma unroll
  for(int k = 1; k < MIN(BLOCK_SIZE, Y_REAL); k++)
  {
   //m = max(m, AS[ty][k]);
   if(m < AS[ty][k])
   {
    m = AS[ty][k];
    im = IS[ty][k];
   }
  }
  
  MS[ty] = m;
  max_idx[i_sample] = (itype)im;
 }
 // ensure max computed
 barrier(CLK_LOCAL_MEM_FENCE);
 m = MS[ty];
 
 dtype sum = 0.0f;
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
