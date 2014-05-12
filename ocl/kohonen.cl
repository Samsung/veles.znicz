#include "defines.cl"
#include "highlight.cl"

/// @brief Kohonen forward propagation.
/// @param h input.
/// @param weights weights.
/// @param y output.
/// @details y = W * h.
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global c_dtype    /* IN */    *h,
                __global c_dtype    /* IN */    *weights,
                __global c_dtype   /* OUT */    *y) {
  #define A_WIDTH BATCH
  #define B_WIDTH Y
  #define AB_COMMON H

  #define A h
  #define B weights

  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #include "matrix_multiplication.cl"

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  if (valid) {
    y[idx] = sum[0];
  }
}


/// @brief Computes distances between input and neuron weights.
/// @param h input.
/// @param weights weights.
/// @param y distance(h, weights).
/// @details Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void compute_distance(__global const c_dtype    /* IN */    *h,
                      __global const c_dtype    /* IN */    *weights,
                      __global dtype           /* OUT */    *y) {
  #define A_WIDTH BATCH
  #define B_WIDTH Y
  #define AB_COMMON H

  #define A h
  #define B weights

  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #define MULTIPLY c_dist2

  #include "matrix_multiplication.cl"

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  if (valid) {
    y[idx] = sum[0];
  }
}


/// @brief Kohonen train pass.
/// @param y Values to find minimum of.
/// @param argmin Indexes of min elements.
/// @details Should be defined externally:
///          REDUCE_SIZE - size for reduction,
///          Y - output size.
#ifdef REDUCE_SIZE
__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void compute_argmin(__global const dtype    /* IN */    *y,
                    __global int            /* OUT */   *argmin) {

  __local dtype AS[REDUCE_SIZE];
  __local int IS[REDUCE_SIZE];

  int bx = get_group_id(0); // from 0 to number of resulting output elements
  int tx = get_local_id(0); // from 0 to REDUCE_SIZE - 1

  dtype min_vle = MAXFLOAT;
  int min_idx = 0;

  int offs = bx * Y + tx;
  int idx = tx;
  for (int i = 0; i < Y / REDUCE_SIZE; i++, offs += REDUCE_SIZE, idx += REDUCE_SIZE) {
    dtype vle = y[offs];
    if (vle < min_vle) {
      min_vle = vle;
      min_idx = idx;
    }
  }
  // Process the remaining part
  #if (Y % REDUCE_SIZE) != 0
  if (tx < Y % REDUCE_SIZE) {
    vle = y[offs];
    if (vle < min_vle) {
      min_vle = vle;
      min_idx = idx;
    }
  }
  #endif

  AS[tx] = min_vle;
  IS[tx] = min_idx;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  // Final reduction
  int n = MIN(Y, REDUCE_SIZE);
  while (n > 1) {
    if ((n & 1) && (AS[n - 1] < min_vle)) {
      min_vle = AS[n - 1];
      min_idx = IS[n - 1];
    }
    n >>= 1;
    if (tx < n) {
      if (AS[n + tx] < AS[tx]) {
        AS[tx] = AS[n + tx];
        IS[tx] = IS[n + tx];
      }
    }
    // ensure all shared summed
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tx == 0) {
    argmin[bx] = min_idx;
  }
}
#endif


/// @brief Computes gravity function from argmin neuron to all others.
/// @param argmin Indexes of neurons with min distances to inputs.
/// @param coords Neuron coordinates in Euclidian space.
/// @param gravity Output gravity.
/// @param sigma Effective radius.
/// @details Should be defined externally:
///          Y - output size,
///          coord_type - type for coordinates of neuron in space (float2).
#ifdef coord_type
__kernel
void compute_gravity(__global const int           /* IN */    *argmin,
                     __global const coord_type    /* IN */    *coords,
                     __global dtype              /* OUT */    *gravity,
                     const dtype                  /* IN */    sigma) {
  int src = get_global_id(0);
  int dst = get_global_id(1);
  dtype d = distance(coords[argmin[src]], coords[dst]);
  gravity[src * Y + dst] = exp((d * d) / (-2 * sigma * sigma));
}
#endif


/// @brief Computes gradients for weights.
/// @param h input.
/// @param weights Weights.
/// @param gradients Gradients to be computed.
/// @details Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          H - input size,
///          Y - output size.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void compute_gradient(__global const c_dtype    /* IN */    *h,
                      __global const c_dtype    /* IN */    *weights,
                      __global c_dtype         /* OUT */    *gradients) {

  #define A h
  #define B weights
  #define A_WIDTH BATCH
  #define B_WIDTH Y
  #define AB_COMMON H
  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #ifdef ALIGNED
  #undef ALIGNED
  #endif
  #if (AB_COMMON % BLOCK_SIZE) == 0
  #define N_BLOCKS (AB_COMMON / BLOCK_SIZE)
  #if ((A_WIDTH % BLOCK_SIZE) == 0) && ((B_WIDTH % BLOCK_SIZE) == 0)
  #define ALIGNED
  #endif
  #else
  #define N_BLOCKS (AB_COMMON / BLOCK_SIZE + 1)
  #endif

  #ifndef A_REAL_OFFS
  #define A_REAL_OFFS a_offs
  #endif
  #ifndef B_REAL_OFFS
  #define B_REAL_OFFS b_offs
  #endif
  #ifndef A_REAL_OFFS_VALID
  #define A_REAL_OFFS_VALID 1
  #endif
  #ifndef B_REAL_OFFS_VALID
  #define B_REAL_OFFS_VALID 1
  #endif

  __local c_dtype AS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of A
  __local c_dtype BS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of B
  __local c_dtype CS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of C

  // Block index in matrix C, where the values will be put
  int bx = get_group_id(0); // from 0 to B_WIDTH / BLOCK_SIZE - 1
  int by = get_group_id(1); // from 0 to A_WIDTH / BLOCK_SIZE - 1

  // Thread index, each thread calculates one element of the resulted submatrix
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
  int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1

  #define A_LIMIT (A_WIDTH * AB_COMMON)
  #ifdef A_COL
  // Block will slide vertically
  int a_x = by * BLOCK_SIZE + tx;
  int a_offs = ty * A_WIDTH + a_x;
  #define A_OFFS A_WIDTH * BLOCK_SIZE
  #define A_INC_X 0
  #define A_LIMIT_X A_WIDTH
  #else
  // Block will slide horizontally
  int a_x = tx;
  int a_offs = (by * BLOCK_SIZE + ty) * AB_COMMON + a_x;
  #define A_OFFS BLOCK_SIZE
  #define A_INC_X BLOCK_SIZE
  #define A_LIMIT_X AB_COMMON
  #endif

  #define B_LIMIT (B_WIDTH * AB_COMMON)
  #ifdef B_COL
  // Block will slide vertically
  int b_x = bx * BLOCK_SIZE + tx;
  int b_offs = ty * B_WIDTH + b_x;
  #define B_OFFS B_WIDTH * BLOCK_SIZE
  #define B_INC_X 0
  #define B_LIMIT_X B_WIDTH
  #else
  // Block will slide horizontally
  int b_x = tx;
  int b_offs = (bx * BLOCK_SIZE + ty) * AB_COMMON + b_x;
  #define B_OFFS BLOCK_SIZE
  #define B_INC_X BLOCK_SIZE
  #define B_LIMIT_X AB_COMMON
  #endif

  for (int i = 0; i < N_BLOCKS; i++, a_offs += A_OFFS, b_offs += B_OFFS) {
    #ifdef ALIGNED
    AS[ty][tx] = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
    BS[ty][tx] = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
    #define b_valid B_REAL_OFFS_VALID
    #else
    AS[ty][tx] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
    int b_valid = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X));
    BS[ty][tx] = b_valid ? B[B_REAL_OFFS] : 0;
    a_x += A_INC_X;
    b_x += B_INC_X;
    #endif
    CS[ty][tx] = 0;

    // ensure all shared loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0, kk = ty; k < BLOCK_SIZE; k++, kk = (kk + 1) % BLOCK_SIZE)
    #ifdef B_COL
    #ifdef A_COL
      CS[kk][tx] += AS[kk][ty] - BS[kk][tx];
    #else
      CS[kk][tx] += AS[ty][kk] - BS[kk][tx];
    #endif
    #else
    #ifdef A_COL
      CS[tx][kk] += AS[kk][ty] - BS[tx][kk];
    #else
      CS[tx][kk] += AS[ty][kk] - BS[tx][kk];
    #endif
    #endif

    // ensure we can reload shared with new values
    barrier(CLK_LOCAL_MEM_FENCE);

    // output CS matrix
    if (b_valid) {
      gradients[B_REAL_OFFS] = CS[ty][tx];
    }
  }
}


/// @brief Applies gradient to weights.
/// @param gradients Gradients to be applied.
/// @param weights Weights.
/// @param gravity Gravities to the winner neuron.
/// @param alpha_batch (global_alpha / batch_size).
/// @param alpha_lambda (-global_alpha * global_lambda).
/// @details Should be defined externally:
///          H - input size,
///          Y - output size.
__kernel
void apply_gradient(__global const c_dtype    /* IN */    *gradients,
                    __global c_dtype          /* OUT */   *weights,
                    __global const dtype      /* IN */    *gravity,
                    const dtype               /* IN */    alpha_batch,
                    const dtype               /* IN */    alpha_lambda) {
  int i_neuron = get_global_id(0);
  int i_x = get_global_id(1);
  #ifdef WEIGHTS_TRANSPOSED
  int idx = i_x * Y + i_neuron;
  #else
  int idx = i_neuron * H + i_x;
  #endif
  c_dtype weight = weights[idx];
  c_dtype gd = gradients[idx] * gravity[i_neuron] * alpha_batch + weight * alpha_lambda;
  weights[idx] = weight + gd;
}
