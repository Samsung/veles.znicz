#include "defines.cl"

/* TODO(a.kazantsev): implement properly.
/// @brief For each sample, outputs the distances to the targets.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param y matrix of samples
/// @param t matrix of targets
/// @param distances matrix of distances
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void mse_find_distances(__global c_dtype *y, __global c_dtype *t,
                        __global dtype *distances) {
}


/// @brief For the given distances, outputs the index of the closest target.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param distances matrix of distances to targets.
/// @param indexes vector of indexes of the closest target.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void mse_find_closest(__global dtype *distances,
                      __global itype *indexes) {
}
*/

/// FIXME(a.kazantsev): The following code is very slow.
/// @brief For the given distances, outputs the index of the closest target.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param y matrix of samples
/// @param t matrix of targets
/// @param n_err number of errors.
__kernel
void mse_find_closest(__global c_dtype *y, __global c_dtype *t,
                      __global itype0 *labels, __global volatile itype2 *n_err) {
  int i_sample = get_global_id(0);
  int y_offs = SAMPLE_SIZE * i_sample;
  int t_offs = 0;
  dtype d_min = MAXFLOAT;
  int i_min = 0;
  for (int i = 0; i < N_TARGETS; i++, t_offs += SAMPLE_SIZE) {
    dtype smm = 0;
    for (int j = 0; j < SAMPLE_SIZE; j++) {
      smm += c_norm(y[y_offs + j] - t[t_offs + j]);
    }
    if (smm < d_min) {
      d_min = smm;
      i_min = i;
    }
  }
  if (labels[i_sample] != i_min) {
    atom_inc(n_err);
  }
}
