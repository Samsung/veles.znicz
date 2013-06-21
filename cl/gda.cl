/*
 * Gradient descent with individual alphas.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


/*
	Weights update with individual alphas.
    
	weights = weights * (1.0 - global_lambda * alphas) - err_y * h * r_batch_size * alphas
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update_a(__global dtype *err_y, __global dtype *h, __global dtype *weights,
                      const dtype r_batch_size, const dtype global_lambda,
                      const dtype alpha_inc, const dtype alpha_dec,
                      const dtype alpha_max, const dtype alpha_min,
                      __global dtype *alphas)
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
 
 dtype alpha = alphas[idx]; // load old alpha
 dtype gd = weights[idx] * global_lambda + sum[0] * r_batch_size; // current gradient
 
 // decrease immediatly, increase for the next step
 dtype aa_pre[3] = {alpha_dec, 1.0f, 1.0f};
 int offs = (int)(sign(gd) * sign(alpha)) + 1;
 alpha = max(fabs(alpha) * aa_pre[offs], alpha_min);
 dtype aa_post[3] = {1.0f, 1.0f, alpha_inc};
 alphas[idx] = copysign(min(alpha * aa_post[offs], alpha_max), gd); // store new alpha
 
 weights[idx] -= gd * alpha; // apply new alpha to current gradient
}


/*
	Bias update with individual alphas.
    
	bias += err_y * k
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void bias_update_a(__global dtype *bias, __global dtype *err_y,
				   const dtype r_batch_size,
				   const dtype alpha_inc, const dtype alpha_dec,
				   const dtype alpha_max, const dtype alpha_min,
				   __global dtype *alphas)
{
 __local dtype AS[BLOCK_SIZE][BLOCK_SIZE];
 
 int bx = get_group_id(0); // from 0 to Y / BLOCK_SIZE - 1
 
 int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
 int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1
 
 dtype sum = 0.0f;
 
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
  
  int idx = get_global_id(0);
  
  dtype alpha = alphas[idx]; // load old alpha
  dtype gd = sum * r_batch_size; // current gradient
  
  // decrease immediatly, increase for the next step
  dtype aa_pre[3] = {alpha_dec, 1.0f, 1.0f};
  int offs = (int)(sign(gd) * sign(alpha)) + 1;
  alpha = max(fabs(alpha) * aa_pre[offs], alpha_min);
  dtype aa_post[3] = {1.0f, 1.0f, alpha_inc};
  alphas[idx] = copysign(min(alpha * aa_post[offs], alpha_max), gd); // store new alpha
  
  bias[idx] -= gd * alpha; // apply new alpha to current gradient
 }
}
