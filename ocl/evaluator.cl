/// @brief Evaluate softmax.
/// @author: Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param y output of the last layer with applied softmax.
/// @param max_idx index of maximum element for each sample in batch.
/// @param labels labels for samples in batch.
/// @param err_y output error for backpropagation.
/// @param n_err [0] - n_err.
/// @param confusion_matrix confusion matrix (may be NULL).
/// @param batch_size size of the current batch.
/// @param max_err_y_sum maximum sum of backpropagated gradient norms.
/// @details We will launch a single workgroup here.
///          Should be defined externally:
///          itype - type of sample label (char)
///          itype2 - type of elements for confusion matrix and n_err_skipped (int)
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
void ev_sm(__global c_dtype /*IN*/ *y, __global itype /*IN*/ *max_idx, __global itype /*IN*/ *labels,
           __global c_dtype /*OUT*/ *err_y, __global itype2 /*IO*/ *n_err,
           __global itype2 /*IO*/ *confusion_matrix, __global dtype /*IO*/ *max_err_y_sum,
           const itype2 batch_size) {
  __local itype2 IM[BLOCK_SIZE], IREAL[BLOCK_SIZE];
  __local dtype SM[BLOCK_SIZE];
  int tx = get_local_id(0);
  int i_sample = tx;
  int y_start = i_sample * Y;
  itype2 n_ok = 0;
  dtype _max_err_y_sum = 0;

  // Compute err_y and fill the confusion matrix
  for (int i = 0; i < N_BLOCKS; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE) {
    dtype err_y_sum = 0;
    if (i_sample < batch_size) {
      itype im = max_idx[i_sample];
      itype ireal = labels[i_sample];

      IM[tx] = im;
      IREAL[tx] = ireal;

      if (im == ireal) {
        n_ok++;
      }
      c_dtype vle;
      for (int j = 0; j < ireal; j++) {
        vle = y[y_start + j];
        err_y[y_start + j] = vle;
        err_y_sum += c_norm(vle);
      }

      vle = y[y_start + ireal] - c_from_re(1);
      err_y[y_start + ireal] = vle;
      err_y_sum += c_norm(vle);

      for (int j = ireal + 1; j < Y; j++) {
        vle = y[y_start + j];
        err_y[y_start + j] = vle;
        err_y_sum += c_norm(vle);
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
/// @author: Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param y output of the last layer.
/// @param target target values.
/// @param err_y output error for backpropagation.
/// @param metrics [0] - sum of sample's mse, [1] - max of sample's mse, [2] - min of sample's mse.
/// @param mse sample's mse.
/// @param batch_size size of the current batch.
/// @details We will launch a single workgroup here.
///          Should be defined externally:
///          itype - type of sample label (char)
///          itype2 - type of elements for confusion matrix and n_err_skipped (int)
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
void ev_mse(__global c_dtype /*IN*/ *y, __global c_dtype /*IN*/ *target,
            __global c_dtype /*OUT*/ *err_y, __global c_dtype /*IO*/ *metrics,
            __global dtype /*OUT*/ *mse, const itype2 batch_size) {
  __local dtype SM[BLOCK_SIZE], SM1[BLOCK_SIZE], SM2[BLOCK_SIZE];
  int tx = get_local_id(0);
  int i_sample = tx;
  int y_start = i_sample * Y;
  dtype mse_sum = 0, mse_max = 0, mse_min = MAXFLOAT;
 
  // Compute err_y and fill the confusion matrix
  for (int i = 0; i < N_BLOCKS; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE) {
    if (i_sample < batch_size) {
      c_dtype vle, vle_target;
      dtype sample_sse = 0;
      for (int j = 0; j < Y; j++) {
        vle = y[y_start + j];
        vle_target = target[y_start + j];
        vle -= vle_target;
        sample_sse += c_norm2(vle);
        err_y[y_start + j] = vle;
      }
      dtype sample_mse = sqrt(sample_sse) / Y;
      mse[i_sample] = sample_mse;
      mse_sum += sample_mse;
      mse_max = max(mse_max, sample_mse);
      mse_min = min(mse_min, sample_mse);
    } else if (i_sample < BATCH) {
      for (int j = 0; j < Y; j++)
        err_y[y_start + j] = 0;
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
