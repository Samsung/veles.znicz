#ifndef _WEIGHTS_ORTO_
#define _WEIGHTS_ORTO_

/// @brief Sums weights matrix over the columns.
/// @param weights weights.
/// @param col_sums output.
/// @details Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void compute_col_sums(__global const dtype    /* IN */    *weights,
                      __global dtype         /* OUT */    *col_sums) {

  #define A weights
#if WEIGHTS_TRANSPOSED > 0
  #define A_WIDTH Y
  #define A_HEIGHT H
#else
  #define A_WIDTH H
  #define A_HEIGHT Y
  #define A_COL
#endif

  #include "matrix_reduce.cl"

#if !(WEIGHTS_TRANSPOSED > 0)
  #undef A_COL
#endif
  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  if (!tx) {
    col_sums[bx] = sum + AS[0];
  }
}

#define gradient_step_ortho(weight, factor, col, n_rows, col_sums) (factor / n_rows * (col_sums[col] - weight))

#endif  // _WEIGHTS_ORTO_
