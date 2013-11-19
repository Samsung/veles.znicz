/* @brief Kernels for convolutional layer gradient descent.
 * @author Kazantsev Alexey <a.kazantsev@samsung.com>
 * @details Should be defined externally:
 *          BLOCK_SIZE - size of the block for matrix multiplication,
 *          BATCH - minibatch size,
 *          SX - input image width,
 *          SY - input image height,
 *          N_CHANNELS - number of input channels,
 *          KX - kernel width,
 *          KY - kernel height,
 *          N_KERNELS - number of kernels (i.e. neurons),
 *          REDUCE_SIZE - buffer size for reduce operation.
 */


#define KERNEL_SIZE (KX * KY * N_CHANNELS)


/// @brief Sums err_h_tmp into err_h.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void err_h_reduce(__global c_dtype /*IN*/ *err_h_tmp, __global c_dtype /*OUT*/ *err_h) {
  __local c_dtype AS[REDUCE_SIZE];

  int bx = get_group_id(0); // from 0 to number of resulting output pixels
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1

  c_dtype sum = c_from_re(0);

  int offs = bx * KERNEL_SIZE + tx;
  for (int i = 0; i < KERNEL_SIZE / REDUCE_SIZE; i++, offs += REDUCE_SIZE) {
    sum += err_h_tmp[offs];
  }
  // Sum the remaining part
  #if (KERNEL_SIZE % REDUCE_SIZE) != 0
  if (tx < KERNEL_SIZE % REDUCE_SIZE)
    sum += err_h_tmp[offs];
  #endif

  AS[tx] = sum;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  // Final summation
  sum = c_from_re(0);
  int n = MIN(KERNEL_SIZE, REDUCE_SIZE);
  while (n > 1) {
    sum += (n & 1) ? AS[n - 1] : c_from_re(0);
    n >>= 1;
    if (tx < n) {
      AS[tx] += AS[n + tx];
    }
    // ensure all shared summed
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (!tx) {
    sum += AS[0];
    err_h[bx] = sum;
  }
}
