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


/*
    Evaluate softmax.

    Parameters:
        y: output of the last layer with applied softmax.
        max_idx: index of maximum element for each sample in batch.
        labels: labels for samples in batch.
        err_y: output error for backpropagation.
        skipped: was sample skipped on previous iteration?
        n_err_skipped: [0] - n_err, [1] - n_skipped.
        confusion_matrix: confusion matrix.
        batch_size: size of the current batch.
        threshold: threshold for computing of skipped.
        threshold_low: threshold low boundary for computing of skipped.

    We will launch single workgroup here.
*/
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, 1, 1)))
void ev_sm(__global dtype /*IN*/ *y, __global itype /*IN*/ *max_idx, __global itype /*IN*/ *labels,
           __global dtype /*OUT*/ *err_y,
           __global uchar /*IO*/ *skipped, __global itype2 /*IO*/ *n_err_skipped,
           __global itype2 /*IO*/ *confusion_matrix,
           const itype2 batch_size, const dtype threshold, const dtype threshold_low)
{
 __local itype2 IM[BLOCK_SIZE], IREAL[BLOCK_SIZE];
 int tx = get_local_id(0);
 int i_sample = tx;
 int y_start = i_sample * Y;
 itype2 n_ok = 0, n_skipped = 0;
 
 // Compute err_y and fill the confusion matrix
 for(int i = 0; i < BATCH / BLOCK_SIZE; i++, i_sample += BLOCK_SIZE, y_start += Y * BLOCK_SIZE)
 {
  if(i_sample < batch_size)
  {
   itype im = max_idx[i_sample];
   itype ireal = labels[i_sample];
   
   IM[tx] = im;
   IREAL[tx] = ireal;
   
   bool skip = false;
   if(im == ireal)
   {
    n_ok++;
    dtype max_vle = y[y_start + im];
    if((max_vle > threshold) || ((max_vle > threshold_low) && (skipped[i_sample])))
    {
     skipped[i_sample] = 1;
     n_skipped++;
     skip = true;
    }
   }
   else
    skipped[i_sample] = 0;
   if(!skip)
   {
    for(int j = 0; j < ireal; j++)
     err_y[y_start + j] = y[y_start + j];
    err_y[y_start + ireal] = y[y_start + ireal] - (dtype)1.0;
    for(int j = ireal + 1; j < Y_REAL; j++)
     err_y[y_start + j] = y[y_start + j];
   }
   else
   {
    for(int j = 0; j < Y_REAL; j++)
     err_y[y_start + j] = 0;
   }
  }
  else
  {
   for(int j = 0; j < Y_REAL; j++)
    err_y[y_start + j] = 0;
  }
  
  // Update confusion matrix  
  barrier(CLK_LOCAL_MEM_FENCE);
  if((!tx) && (confusion_matrix) && (i_sample < batch_size))
  {
   int n = batch_size - i_sample;
   if(n > BLOCK_SIZE)
    n = BLOCK_SIZE;
   for(int j = 0; j < n; j++)
    confusion_matrix[IM[j] * Y_REAL + IREAL[j]]++;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
 }
 
 // Compute total n_ok and n_skipped
 IM[tx] = n_ok;
 IREAL[tx] = n_skipped;
 barrier(CLK_LOCAL_MEM_FENCE);
 if(!tx)
 {
  n_ok = IM[0];
  n_skipped = IREAL[0];
  for(int j = 1; j < BLOCK_SIZE; j++)
  {
   n_ok += IM[j];
   n_skipped += IREAL[j];
  }
  n_err_skipped[0] += batch_size - n_ok;
  n_err_skipped[1] += n_skipped;
 }
}
