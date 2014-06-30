#include "defines.cl"
#include "highlight.cl"


/// @brief Computes softmax and finds element with maximum real part.
/// @details For each sample from batch of y:
///          1. Find m = max().
///          2. Overwrite x as exp(x - m).
///          3. Compute sum of all x.
///          4. Divide x by sum.
///          Should be defined externally:
///          REDUCE_SIZE - size for reduction,
///          BATCH - minibatch size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void apply_exp(__global dtype *y, __global int *max_idx) {
  __local dtype AS[REDUCE_SIZE];
  __local int IS[REDUCE_SIZE];

  int bx = get_group_id(0); // from 0 to number of resulting output elements
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1


  // 1. Find the maximum element

  dtype max_vle = -MAXFLOAT;
  int max_vle_idx = 0;

  int offs = bx * Y + tx, i_offs = tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE, i_offs += REDUCE_SIZE) {
    if (max_vle < y[offs]) {
      max_vle = y[offs];
      max_vle_idx = i_offs;
    }
  }
  // Process the remaining part
  #if (Y % REDUCE_SIZE) != 0
  if (tx < Y % REDUCE_SIZE) {
    if (max_vle < y[offs]) {
      max_vle = y[offs];
      max_vle_idx = i_offs;
    }
  }
  #endif

  AS[tx] = max_vle;
  IS[tx] = max_vle_idx;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  // Process found elements
  int n = MIN(Y, REDUCE_SIZE);
  while (n > 1) {
    if (n & 1) {
      if (max_vle < AS[n - 1]) {
        max_vle = AS[n - 1];
        max_vle_idx = IS[n - 1];
      }
    }
    n >>= 1;
    if (tx < n) {
      if (AS[tx] < AS[n + tx]) {
        AS[tx] = AS[n + tx];
        IS[tx] = IS[n + tx];
      }
    }
    // ensure all shared updated
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (!tx) {
    if (AS[0] < max_vle) {
      AS[0] = max_vle;
      IS[0] = max_vle_idx;
    }
    max_idx[bx] = IS[0];
  }
  // ensure all shared updated
  barrier(CLK_LOCAL_MEM_FENCE);

  max_vle = AS[0];

  // ensure all shared read 'cause we will update AS later
  barrier(CLK_LOCAL_MEM_FENCE);

  // 2. Find the sum(exp(x - max))
  dtype sum = 0;

  offs = bx * Y + tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE) {
    sum += exp(y[offs] - max_vle);
  }
  // Process the remaining part
  #if (Y % REDUCE_SIZE) != 0
  if (tx < Y % REDUCE_SIZE) {
    sum += exp(y[offs] - max_vle);
  }
  #endif

  AS[tx] = sum;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  // Process found elements
  sum = 0;
  n = MIN(Y, REDUCE_SIZE);
  while (n > 1) {
    if (n & 1) {
      sum += AS[n - 1];
    }
    n >>= 1;
    if (tx < n) {
      AS[tx] += AS[n + tx];
    }
    // ensure all shared summed
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (!tx) {
    AS[0] += sum;
  }
  // ensure all shared updated
  barrier(CLK_LOCAL_MEM_FENCE);

  sum = AS[0];


  // 3. Output exp(x - max) / sum
  offs = bx * Y + tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE) {
    y[offs] = exp(y[offs] - max_vle) / sum;
  }
  // Process the remaining part
  #if (Y % REDUCE_SIZE) != 0
  if (tx < Y % REDUCE_SIZE) {
    y[offs] = exp(y[offs] - max_vle) / sum;
  }
  #endif
}
