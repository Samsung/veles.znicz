#include "defines.cl"
#include "highlight.cl"

#ifndef INCLUDE_BIAS
#error "INCLUDE_BIAS should be defined"
#endif

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

#include "conv_common.cl"

/// @brief Feeds 2D multichannel convolutional layer with activation function:
///        linear activation: x;
///        scaled tanh activation: 1.7159 * tanh(0.6666 * x),
///        because: f(1) = 1, f(-1) = -1 and f"(x) maximum at x = 1.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param input batch of input multichannel interleaved images.
/// @param weights weights (matrix of size (KX * KY * N_CHANNELS, N_KERNELS)).
/// @param output batch of output multichannel interleaved images.
/// @param bias bias (vector of length N_KERNELS).
/// @details y = f(input * weights + bias), where input is the combined matrix.
///          Will do convolution via matrix multiplication
///          ('cause we have to convolve the "batch" with "multiple" kernels at once).
///          The process we are doing here equal to the convolution with reflected weights matrix,
///          and it is so in favor of the nature of the process of applying neurons (i.e. kernels) to each point of the input image.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global const dtype    /* IN */    *input,
                __global const dtype    /* IN */    *weights,
                #if INCLUDE_BIAS > 0
                __global dtype          /* IN */    *bias,
                #endif
                __global dtype         /* OUT */    *output) {

  #define A_WIDTH (BATCH * KERNELS_PER_SAMPLE)
  #define B_WIDTH N_KERNELS
  #define AB_COMMON (KX * KY * N_CHANNELS)

  #define A input
  #define B weights

  #if WEIGHTS_TRANSPOSED > 0
  #define B_COL
  #endif

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS
  #define A_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #define STORE_OUTPUT "conv/forward.store_output.cl"
  #include "matrix_multiplication.cl"

  #undef in_offs
  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B
}
