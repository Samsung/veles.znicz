#include "defines.cl"
#include "highlight.cl"

/// @brief Computes softmax and finds element with maximum real part.
/// @details For each sample from batch of y:
///          1. Find m = max().
///          2. Overwrite x as exp(x - m).
///          3. Compute sum of all x.
///          4. Divide x by sum.
///          Should be defined externally:
///          BLOCK_SIZE - block size,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void apply_exp(__global c_dtype *y, __global int *max_idx) {
  __local c_dtype AS[BLOCK_SIZE];
  __local int IS[BLOCK_SIZE];

  int bx = get_group_id(0); // from 0 to BATCH / BLOCK_SIZE - 1
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1

  int i_sample = bx;
  int sample_offs = i_sample * Y;
  int start_offs = sample_offs + tx;

  c_dtype m = c_from_re(-MAXFLOAT);
  int im = 0;
  int offs = start_offs;
  for (int i = 0; i < Y / BLOCK_SIZE; i++, offs += BLOCK_SIZE) {
    //m = max(m, y[offs]);
    c_dtype vle = y[offs];
    if (c_re(m) < c_re(vle)) {
      m = vle;
      im = offs - sample_offs;
    }
  }
  // Process the remaining part
  #if (Y % BLOCK_SIZE) != 0
  if (tx < Y % BLOCK_SIZE) {
    //m = max(m, y[offs]);
    c_dtype vle = y[offs];
    if (c_re(m) < c_re(vle)) {
      m = vle;
      im = offs - sample_offs;
    }
  }
  #endif
  AS[tx] = m;
  IS[tx] = im;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  if (!tx) {
    m = AS[0];
    im = IS[0];

    #pragma unroll
    for (int k = 1; k < MIN(BLOCK_SIZE, Y); k++) {
      //m = max(m, AS[k]);
      if (c_re(m) < c_re(AS[k])) {
        m = AS[k];
        im = IS[k];
      }
    }

    AS[0] = m;
    max_idx[i_sample] = im; // output found maximum element index
  }
  // ensure max computed
  barrier(CLK_LOCAL_MEM_FENCE);
  m = AS[0];

  c_dtype sum = c_from_re(0);
  offs = start_offs;
  for (int i = 0; i < Y / BLOCK_SIZE; i++, offs += BLOCK_SIZE)
    sum += c_exp(y[offs] - m);
  #if (Y % BLOCK_SIZE) != 0
  if (tx < Y % BLOCK_SIZE)
    sum += c_exp(y[offs] - m);
  #endif
  AS[tx] = sum;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  if (!tx) {
    sum = AS[0];

    #pragma unroll
    for (int k = 1; k < MIN(BLOCK_SIZE, Y); k++)
      sum += AS[k];

    AS[0] = sum;
  }
  // ensure sum computed
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = AS[0];

  offs = start_offs;
  for (int i = 0; i < Y / BLOCK_SIZE; i++, offs += BLOCK_SIZE) {
    y[offs] = c_div(c_exp(y[offs] - m), sum);
  }
  #if (Y % BLOCK_SIZE) != 0
  if (tx < Y % BLOCK_SIZE)
    y[offs] = c_div(c_exp(y[offs] - m), sum);
  #endif
}
