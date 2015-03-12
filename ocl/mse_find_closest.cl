#include "defines.cl"
#include "highlight.cl"

/* TODO(a.kazantsev): implement properly.
/// @brief For each sample, outputs the distances to the targets.
/// @param y matrix of samples
/// @param t matrix of targets
/// @param distances matrix of distances
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void mse_find_distances(__global dtype *y, __global dtype *t,
                        __global dtype *distances) {
}


/// @brief For the given distances, outputs the index of the closest target.
/// @param distances matrix of distances to targets.
/// @param indexes vector of indexes of the closest target.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void mse_find_closest(__global dtype *distances,
                      __global int *indexes) {
}
*/

/// FIXME(a.kazantsev): The following code is very slow.
/// @brief For the given distances, outputs the index of the closest target.
/// @param y matrix of samples
/// @param t matrix of targets
/// @param n_err number of errors.
__kernel
void mse_find_closest(__global const dtype *y, __global const target_dtype *t,
                      __global const int *labels, __global volatile int *n_err) {
  int i_sample = get_global_id(0);
  int y_offs = SAMPLE_SIZE * i_sample;
  int t_offs = 0;
  dtype d_min = MAXFLOAT;
  int i_min = 0;
  for (int i = 0; i < N_TARGETS; i++, t_offs += SAMPLE_SIZE) {
    dtype smm = 0;
    for (int j = 0; j < SAMPLE_SIZE; j++) {
      dtype vle = y[y_offs + j] - t[t_offs + j];
      smm += vle * vle;
    }
    if (smm < d_min) {
      d_min = smm;
      i_min = i;
    }
  }
  if (labels[i_sample] != i_min) {
  	atomic_inc(n_err);
  }
}
