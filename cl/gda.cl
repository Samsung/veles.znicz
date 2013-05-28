/*
	Weights update with individual alphas.
    
	weights = weights * (1.0 - global_lambda * alphas) - err_y * h * r_batch_size * alphas
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update_a(__global float *err_y, __global float *h, __global float *weights,
                      const float r_batch_size, const float global_lambda,
                      const float alpha_inc, const float alpha_dec,
                      const float alpha_max, const float alpha_min,
                      __global float *alphas)
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
 
 float alpha = alphas[idx]; // load old alpha
 float gd = weights[idx] * global_lambda + sum[0] * r_batch_size; // current gradient
 
 // decrease immediatly, increase for the next step
 float aa_pre[3] = {alpha_dec, 1.0f, 1.0f};
 int offs = (int)(sign(gd) * sign(alpha)) + 1;
 alpha = max(fabs(alpha) * aa_pre[offs], alpha_min);
 float aa_post[3] = {1.0f, 1.0f, alpha_inc};
 alphas[idx] = copysign(min(alpha * aa_post[offs], alpha_max), gd); // store new alpha
 
 weights[idx] -= gd * alpha; // apply new alpha to current gradient
}


/*
	Bias update with individual alphas.
    
	bias += err_y * k
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void bias_update_a(__global float *bias, __global float *err_y,
				   const float r_batch_size,
				   const float alpha_inc, const float alpha_dec,
				   const float alpha_max, const float alpha_min,
				   __global float *alphas)
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
  
  int idx = get_global_id(0);
  
  float alpha = alphas[idx]; // load old alpha
  float gd = sum * r_batch_size; // current gradient
  
  // decrease immediatly, increase for the next step
  float aa_pre[3] = {alpha_dec, 1.0f, 1.0f};
  int offs = (int)(sign(gd) * sign(alpha)) + 1;
  alpha = max(fabs(alpha) * aa_pre[offs], alpha_min);
  float aa_post[3] = {1.0f, 1.0f, alpha_inc};
  alphas[idx] = copysign(min(alpha * aa_post[offs], alpha_max), gd); // store new alpha
  
  bias[idx] -= gd * alpha; // apply new alpha to current gradient
 }
}
