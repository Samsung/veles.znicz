/*
 * Evaluates output:
 *   - must compute error for backpropagation,
 *   - may compute number of wrong-guessed samples and confusion matrix.
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
 */


//Should be declared externally:
//#define dtype float /* floating point number type */
//#define itype uchar /* type of sample label */
//#define itype2 uint /* type of elements for confusion matrix and n_err_skipped */
//#define BLOCK_SIZE 30 /* BATCH should be multiple of BLOCK_SIZE */
//#define BATCH 600 /* total batch size */
//#define Y 24 /* last layer output size aligned to BLOCK_SIZE */
//#define Y_REAL 5 /* real last layer output size */


/// @brief Evaluate softmax.
/// @param y output of the last layer with applied softmax.
/// @param max_idx index of maximum element for each sample in batch.
/// @param labels labels for samples in batch.
/// @param err_y output error for backpropagation.
/// @param skipped was sample skipped on previous iteration?
/// @param n_err_skipped [0] - n_err, [1] - n_skipped.
/// @param confusion_matrix confusion matrix (may be NULL).
/// @param batch_size size of the current batch.
/// @param threshold threshold for computing of skipped.
/// @param threshold_low threshold low boundary for computing of skipped.
/// @param max_err_y_sum maximum backpropagated gradient.
/// @details We will launch a single workgroup here.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void ev_sm(__global dtype /*IN*/ *y, __global itype /*IN*/ *max_idx, __global itype /*IN*/ *labels,
           __global dtype /*OUT*/ *err_y,
           __global uchar /*IO*/ *skipped, __global itype2 /*IO*/ *n_err_skipped,
           __global itype2 /*IO*/ *confusion_matrix, __global dtype /*IO*/ *max_err_y_sum,
           const itype2 batch_size, const dtype threshold, const dtype threshold_low) {
  __local itype2 IM[BLOCK_SIZE], IREAL[BLOCK_SIZE];
  __local dtype SM[BLOCK_SIZE];
  int tx = get_local_id(0);
  int i_sample = tx;
  int y_start = i_sample * Y;
  itype2 n_ok = 0, n_skipped = 0;
  dtype err_y_sum = 0;
 
  // Compute err_y and fill the confusion matrix
  for (int i = 0; i < BATCH / BLOCK_SIZE; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE) {
    if (i_sample < batch_size) {
      itype im = max_idx[i_sample];
      itype ireal = labels[i_sample];
   
      IM[tx] = im;
      IREAL[tx] = ireal;
   
      bool skip = false;
      if (im == ireal) {
        n_ok++;
        dtype max_vle = y[y_start + im];
        if ((max_vle > threshold) || ((max_vle > threshold_low) && (skipped[i_sample]))) {
          skipped[i_sample] = 1;
          n_skipped++;
          skip = true;
        }
      } else {
        skipped[i_sample] = 0;
      }
      if (!skip) {
        dtype vle;
        for (int j = 0; j < ireal; j++) {
          vle = y[y_start + j];
          err_y[y_start + j] = vle;
          err_y_sum += fabs(vle);
        }
        vle = y[y_start + ireal] - (dtype)1.0;
        err_y[y_start + ireal] = vle;
        err_y_sum += fabs(vle);
        for (int j = ireal + 1; j < Y_REAL; j++) {
          vle = y[y_start + j];
          err_y[y_start + j] = vle;
          err_y_sum += fabs(vle);
        }
      } else {
        for (int j = 0; j < Y_REAL; j++)
          err_y[y_start + j] = 0;
      }
    } else {
      for (int j = 0; j < Y_REAL; j++)
        err_y[y_start + j] = 0;
    }
  
    // Update confusion matrix
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((!tx) && (confusion_matrix) && (i_sample < batch_size)) {
      int n = batch_size - i_sample;
      if (n > BLOCK_SIZE)
        n = BLOCK_SIZE;
      for (int j = 0; j < n; j++)
        confusion_matrix[IM[j] * Y_REAL + IREAL[j]]++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
 
  // Compute total n_ok and n_skipped
  IM[tx] = n_ok;
  IREAL[tx] = n_skipped;
  SM[tx] = err_y_sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!tx) {
    n_ok = IM[0];
    n_skipped = IREAL[0];
    err_y_sum = SM[tx];
    for (int j = 1; j < BLOCK_SIZE; j++) {
      n_ok += IM[j];
      n_skipped += IREAL[j];
      err_y_sum = max(err_y_sum, SM[j]);
    }
    n_err_skipped[0] += batch_size - n_ok;
    n_err_skipped[1] += n_skipped;
    max_err_y_sum[0] = max(err_y_sum, max_err_y_sum[0]);
  }
}


/// @brief Evaluate MSE.
/// @param y output of the last layer with applied softmax.
/// @param max_idx index of maximum element for each sample in batch.
/// @param labels labels for samples in batch.
/// @param err_y output error for backpropagation.
/// @param n_err_skipped [0] - n_err (pointwise), [1] - n_skipped (pointwise).
/// @param batch_size size of the current batch
/// @param threshold_skip when difference between output and target becomes lower
///                       than this value, assume gradient as 0.
/// @param threshold_ok when mse between output and target becomes lower
///                       than this value, icrement n_ok.
/// @param metrics [0] - sum of sample's mse, [1] - max of sample's mse, [2] - min of sample's mse.
/// @param mse sample's mse (may be NULL).
/// @details We will launch a single workgroup here.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void ev_mse(__global dtype /*IN*/ *y, __global dtype /*IN*/ *target,
            __global dtype /*OUT*/ *err_y, __global itype2 /*IO*/ *n_err_skipped,
            __global dtype /*IO*/ *metrics, const itype2 batch_size,
            const dtype threshold_skip, const dtype threshold_ok,
            __global dtype /*OUT*/ *mse) {
  __local itype2 N_OK[BLOCK_SIZE], N_SKIP[BLOCK_SIZE];
  __local dtype SM[BLOCK_SIZE], SM1[BLOCK_SIZE], SM2[BLOCK_SIZE];
  int tx = get_local_id(0);
  int i_sample = tx;
  int y_start = i_sample * Y;
  itype2 n_ok = 0, n_skipped = 0;
  dtype mse_sum = 0, mse_max = 0, mse_min = (dtype)1.0e30;
 
  // Compute err_y and fill the confusion matrix
  for (int i = 0; i < BATCH / BLOCK_SIZE; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE) {
    if (i_sample < batch_size) {
      dtype vle, vle_target, vle_diff, vle_diffa, sample_sse = 0;
      for (int j = 0; j < Y_REAL; j++) {
        vle = y[y_start + j];
        vle_target = target[y_start + j];
        vle -= vle_target;
        sample_sse += vle * vle;
        err_y[y_start + j] = vle;
      }
      dtype sample_mse = sqrt(sample_sse) / Y_REAL;
      if (mse)
        mse[i_sample] = sample_mse;
      if (sample_mse < threshold_ok)
        n_ok++;
      if (sample_mse < threshold_skip) {
        for (int j = 0; j < Y_REAL; j++)
          err_y[y_start + j] = 0;
      }
      mse_sum += sample_mse;
      mse_max = max(mse_max, sample_mse);
      mse_min = min(mse_min, sample_mse);
    } else {
      for (int j = 0; j < Y_REAL; j++)
        err_y[y_start + j] = 0;
    }
  }
  // Compute total n_ok and n_skipped
  N_OK[tx] = n_ok;
  N_SKIP[tx] = n_skipped;
  SM[tx] = mse_sum;
  SM1[tx] = mse_max;
  SM2[tx] = mse_min;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!tx) {
    n_ok = N_OK[0];
    n_skipped = N_SKIP[0];
    mse_sum = SM[tx];
    mse_max = SM1[tx];
    mse_min = SM2[tx];
    for (int j = 1; j < BLOCK_SIZE; j++) {
      n_ok += N_OK[j];
      n_skipped += N_SKIP[j];
      mse_sum += SM[j];
      mse_max = max(mse_max, SM1[j]);
      mse_min = min(mse_min, SM2[j]);
    }
    n_err_skipped[0] += batch_size - n_ok;
    n_err_skipped[1] += n_skipped;
    metrics[0] += mse_sum;
    metrics[1] = max(metrics[1], mse_max);
    metrics[2] = min(metrics[2], mse_min);
  }
}
