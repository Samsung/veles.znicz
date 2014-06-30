#include "defines.cl"
#include "highlight.cl"

/// @brief Evaluate softmax.
/// @param y output of the last layer with applied softmax.
/// @param max_idx index of maximum element for each sample in batch.
/// @param labels labels for samples in batch.
/// @param err_y output error for backpropagation.
/// @param n_err [0] - n_err.
/// @param confusion_matrix confusion matrix (may be NULL).
/// @param max_err_y_sum maximum sum of backpropagated gradient norms.
/// @param batch_size size of the current batch.
/// @param multiplier coefficient to multiply backpropagated error on.
/// @details We will launch a single workgroup here.
///          Should be defined externally:
///          BLOCK_SIZE - block size
///          BATCH - minibatch size
///          Y - last layer output size.
#ifdef N_BLOCKS
#undef N_BLOCKS
#endif
#if (BATCH % BLOCK_SIZE) == 0
#define N_BLOCKS (BATCH / BLOCK_SIZE)
#else
#define N_BLOCKS (BATCH / BLOCK_SIZE + 1)
#endif
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void ev_sm(__global const dtype    /* IN */    *y,
           __global const int      /* IN */    *max_idx,
           __global const int      /* IN */    *labels,
           __global dtype         /* OUT */    *err_y,
           __global int       /* IN, OUT */    *n_err,
           __global int       /* IN, OUT */    *confusion_matrix,
           __global dtype     /* IN, OUT */    *max_err_y_sum,
           const int               /* IN */    batch_size,
           const dtype             /* IN */    multiplier) {
  __local int IM[BLOCK_SIZE], IREAL[BLOCK_SIZE];
  __local dtype SM[BLOCK_SIZE];
  int tx = get_local_id(0);
  int i_sample = tx;
  int y_start = i_sample * Y;
  int n_ok = 0;
  dtype _max_err_y_sum = 0;

  // Compute err_y and fill the confusion matrix
  for (int i = 0; i < N_BLOCKS; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE) {
    dtype err_y_sum = 0;
    if (i_sample < batch_size) {
      int im = max_idx[i_sample];
      int ireal = labels[i_sample];

      IM[tx] = im;
      IREAL[tx] = ireal;

      if (im == ireal) {
        n_ok++;
      }
      dtype vle;
      for (int j = 0; j < ireal; j++) {
        vle = y[y_start + j];
        vle *= multiplier;
        err_y[y_start + j] = vle;
        err_y_sum += fabs(vle);
      }

      vle = y[y_start + ireal] - 1;
      vle *= multiplier;
      err_y[y_start + ireal] = vle;
      err_y_sum += fabs(vle);

      for (int j = ireal + 1; j < Y; j++) {
        vle = y[y_start + j];
        vle *= multiplier;
        err_y[y_start + j] = vle;
        err_y_sum += fabs(vle);
      }
    } else if (i_sample < BATCH) { // set excessive gradients to zero
      for (int j = 0; j < Y; j++)
        err_y[y_start + j] = 0;
    }
    _max_err_y_sum = max(_max_err_y_sum, err_y_sum);

    // Update confusion matrix
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((!tx) && (confusion_matrix) && (i_sample < batch_size)) {
      int n = batch_size - i_sample;
      if (n > BLOCK_SIZE)
        n = BLOCK_SIZE;
      for (int j = 0; j < n; j++)
        confusion_matrix[IM[j] * Y + IREAL[j]]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
 
  // Compute n_err, max_err_y_sum
  IM[tx] = n_ok;
  SM[tx] = _max_err_y_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!tx) {
    n_ok = IM[0];
    _max_err_y_sum = SM[tx];
    for (int j = 1; j < BLOCK_SIZE; j++) {
      n_ok += IM[j];
      _max_err_y_sum = max(_max_err_y_sum, SM[j]);
    }
    n_err[0] += batch_size - n_ok;
    max_err_y_sum[0] = max(_max_err_y_sum, max_err_y_sum[0]);
  }
}


/// @brief Evaluate MSE.
/// @param y output of the last layer.
/// @param target target values.
/// @param err_y output error for backpropagation.
/// @param metrics [0] - sum of sample's mse, [1] - max of sample's mse, [2] - min of sample's mse.
/// @param mse sample's mse.
/// @param batch_size size of the current batch.
/// @param multiplier coefficient to multiply backpropagated error on.
/// @details We will launch a single workgroup here.
///          Should be defined externally:
///          BLOCK_SIZE - block size
///          BATCH - minibatch size
///          Y - last layer output size.
#ifdef N_BLOCKS
#undef N_BLOCKS
#endif
#if (BATCH % BLOCK_SIZE) == 0
#define N_BLOCKS (BATCH / BLOCK_SIZE)
#else
#define N_BLOCKS (BATCH / BLOCK_SIZE + 1)
#endif
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void ev_mse(__global const dtype    /* IN */    *y,
            __global const dtype    /* IN */    *target,
            __global dtype         /* OUT */    *err_y,
            __global dtype     /* IN, OUT */    *metrics,
            __global dtype         /* OUT */    *mse,
            const int               /* IN */    batch_size,
            const dtype             /* IN */    multiplier) {
  __local dtype SM[BLOCK_SIZE], SM1[BLOCK_SIZE], SM2[BLOCK_SIZE];
  int tx = get_local_id(0);
  int i_sample = tx;
  int y_start = i_sample * Y;
  dtype mse_sum = 0, mse_max = 0, mse_min = MAXFLOAT;
 
  // Compute err_y and fill the confusion matrix
  for (int i = 0; i < N_BLOCKS; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE) {
    if (i_sample < batch_size) {
      dtype vle, vle_target;
      dtype sample_sse = 0;
      for (int j = 0; j < Y; j++) {
        vle = y[y_start + j];
        vle_target = target[y_start + j];
        vle -= vle_target;
        sample_sse += vle * vle;
        vle *= multiplier;
        err_y[y_start + j] = vle;
      }
      dtype sample_mse = sqrt(sample_sse / Y);
      mse[i_sample] = sample_mse;
      mse_sum += sample_mse;
      mse_max = max(mse_max, sample_mse);
      mse_min = min(mse_min, sample_mse);
    } else if (i_sample < BATCH) {
      for (int j = 0; j < Y; j++) {
        err_y[y_start + j] = 0;
      }
      mse[i_sample] = 0;
    }
  }
  // Compute metrics
  SM[tx] = mse_sum;
  SM1[tx] = mse_max;
  SM2[tx] = mse_min;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!tx) {
    mse_sum = SM[tx];
    mse_max = SM1[tx];
    mse_min = SM2[tx];
    for (int j = 1; j < BLOCK_SIZE; j++) {
      mse_sum += SM[j];
      mse_max = max(mse_max, SM1[j]);
      mse_min = min(mse_min, SM2[j]);
    }
    metrics[0] += mse_sum;
    metrics[1] = max(metrics[1], mse_max);
    metrics[2] = min(metrics[2], mse_min);
  }
}
