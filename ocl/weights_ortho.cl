#ifndef _WEIGHTS_ORTO_
#define _WEIGHTS_ORTO_

/// @brief Computes matrix product of weights matrix with itself transposed.
/// @param weights weights.
/// @param wwt output.
/// @details wwt = dot(W, W.transpose())
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void compute_wwt(__global const dtype    /* IN */    *weights,
                 __global dtype         /* OUT */    *wwt) {
  #define A_WIDTH Y
  #define B_WIDTH Y
  #define AB_COMMON H

  #define A weights
  #define B weights

  #if WEIGHTS_TRANSPOSED > 0
  #define A_COL
  #define B_COL
  #endif

  #include "matrix_multiplication.cl"

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  if (valid) {
    wwt[idx] = sum;
  }
}


/// @brief Sums wwt matrix over the rows excluding diagonal element.
/// @param wwt dot(weights, weights.transpose()).
/// @param row_sums output.
/// @details Should be defined externally:
///          REDUCE_SIZE - size of the block for matrix reduce,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void compute_row_sums(__global const dtype    /* IN */    *wwt,
                      __global dtype         /* OUT */    *row_sums) {

  #define A wwt
  #define A_WIDTH Y
  #define A_HEIGHT Y

  #include "matrix_reduce.cl"

  #undef A_HEIGHT
  #undef A_WIDTH
  #undef A

  if (!tx) {
    row_sums[bx] = sum + AS[0] - wwt[bx * Y + bx];
  }
}


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

#define gradient_step_ortho(weight, factor, row, col, row_sums, col_sums) (factor * row_sums[row] * (col_sums[col] - weight))

#endif  // _WEIGHTS_ORTO_
