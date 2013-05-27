/*
	Weights update with individual alphas.
    
	weights = weights * (1.0 - global_lambda * alphas) - err_y * h * r_batch_size * alphas
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update_a(__global float *err_y, __global float *h, __global float *weights,
                      const float r_batch_size, const float global_lambda,
                      const float alpha_inc, const float alpha_dec,
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
 
 float gd = sum[0] * r_batch_size;
 float alpha = alphas[idx];
 float alpha_plus = fabs(alpha);
 
 float aa[3] = {alpha_dec, 1.0f, alpha_inc};
 int offs = (int)(sign(gd) * sign(alpha)) + 1;
 alpha *= aa[offs];
 alphas[idx] = copysign(alpha, gd);
 
 weights[idx] = weights[idx] * (1.0f - global_lambda * alpha_plus) - gd * alpha_plus;
}


/*
	Bias update with individual alphas.
    
	bias += err_y * k
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void bias_update_a(__global float *bias, __global float *err_y,
				   const float r_batch_size,
				   const float alpha_inc, const float alpha_dec,
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
  
  float gd = sum * r_batch_size;
  float alpha = alphas[idx];
  float alpha_plus = fabs(alpha);
  
  float aa[3] = {alpha_dec, 1.0f, alpha_inc};
  int offs = (int)(sign(gd) * sign(alpha)) + 1;
  alpha *= aa[offs];
  alphas[idx] = copysign(alpha, gd);
  
  bias[idx] -= gd * alpha_plus;
 }
}
