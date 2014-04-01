#include "defines.cl"

/// @brief Feeds 2D multichannel convolutional layer with activation function:
///        linear activation: x;
///        scaled tanh activation: 1.7159 * tanh(0.6666 * x),
///        because: f(1) = 1, f(-1) = -1 and f"(x) maximum at x = 1.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @param h batch of input multichannel interleaved images.
/// @param weights weights (matrix of size (KX * KY * N_CHANNELS, N_KERNELS)).
/// @param y batch of output multichannel interleaved images.
/// @param bias bias (vector of length N_KERNELS).
/// @details y = f(input * weights + bias), where input is the combined matrix.
///          Will do convolution via matrix multiplication
///          ('cause we have to convolve the "batch" with "multiple" kernels at once).
///          Should be defined externally:
///          BLOCK_SIZE - size of the block for matrix multiplication,
///          BATCH - minibatch size,
///          SX - input image width,
///          SY - input image height,
///          N_CHANNELS - number of input channels,
///          KX - kernel width,
///          KY - kernel height,
///          N_KERNELS - number of kernels (i.e. neurons).
///          The process we are doing here equal to the convolution with reflected weights matrix,
///          and it is so in favor of the nature of the process of applying neurons (i.e. kernels) to each point of the input image.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global c_dtype /*IN*/ *h, __global c_dtype /*IN*/ *weights,
                __global c_dtype /*OUT*/ *y, __global c_dtype /*IN*/ *bias) {
  #define A_WIDTH (BATCH * (SX - KX + 1) * (SY - KY + 1))
  #define B_WIDTH N_KERNELS
  #define AB_COMMON (KX * KY * N_CHANNELS)

  #define A h
  #define B weights
  #define C y

  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #define ELEMENTS_PER_BLOCK (N_CHANNELS * KX * KY)
  #define BLOCK_NUMBER (a_offs / ELEMENTS_PER_BLOCK)
  #define OFFS_IN_BLOCK (a_offs % ELEMENTS_PER_BLOCK)
  #define ROW_IN_BLOCK (OFFS_IN_BLOCK / (N_CHANNELS * KX))
  #define COL_IN_BLOCK (OFFS_IN_BLOCK % (N_CHANNELS * KX))
  #define SAMPLE_NUMBER (BLOCK_NUMBER / BLOCKS_PER_SAMPLE)
  #define BLOCK_IN_SAMPLE (BLOCK_NUMBER % BLOCKS_PER_SAMPLE)

  #if CONVOLUTION_TYPE_VALID > 0

  #define BLOCKS_PER_SAMPLE ((SX - KX + 1) * (SY - KY + 1))
  #define ROW_IN_SAMPLE (BLOCK_IN_SAMPLE / (SX - KX + 1))
  #define COL_IN_SAMPLE ((BLOCK_IN_SAMPLE % (SX - KX + 1)) * N_CHANNELS)
  #define A_REAL_OFFS ((SAMPLE_NUMBER * SY + ROW_IN_SAMPLE + ROW_IN_BLOCK) * (N_CHANNELS * SX) + (COL_IN_SAMPLE + COL_IN_BLOCK))

  #elif CONVOLUTION_TYPE_SAME > 0

  #define BLOCKS_PER_SAMPLE (SX * SY)

  #if (KX - 1) % 2 == 0
  #define SHIFT_LEFT ((KX - 1) / 2)
  #else
  #define SHIFT_LEFT ((KX - 1) / 2 + 1)
  #endif

  #if (KY - 1) % 2 == 0
  #define SHIFT_TOP ((KY - 1) / 2)
  #else
  #define SHIFT_TOP ((KY - 1) / 2 + 1)
  #endif

  #define ROW_IN_SAMPLE (BLOCK_IN_SAMPLE / SX - SHIFT_LEFT)
  #define COL_IN_SAMPLE ((BLOCK_IN_SAMPLE % SX) * N_CHANNELS - SHIFT_TOP)

  #define A_REAL_OFFS_VALID (ROW_IN_SAMPLES >= 0) && (ROW_IN_SAMPLE < SX) && (COL_IN_SAMPLE >= 0) && (COL_IN_SAMPLE < SY)

  #error "TODO(a.kazantsev): implement remaining."
  #define A_REAL_OFFS ((SAMPLE_NUMBER * SY + ROW_IN_SAMPLE + ROW_IN_BLOCK) * (N_CHANNELS * SX) + (COL_IN_SAMPLE + COL_IN_BLOCK))

  #else
  #error "Convolution type should be defined"
  #endif

  #include "matrix_multiplication.cl"

  #undef A_REAL_OFFS

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B
  #undef C

  if ((valid) && (!ty)) // read from memory only for the first row
    AS[0][tx] = bias[bx * BLOCK_SIZE + tx];

  barrier(CLK_LOCAL_MEM_FENCE);

  if (valid) {
    c_dtype s = sum[0] + AS[0][tx];
    #if ACTIVATION_LINEAR > 0
    y[idx] = s;
    #elif ACTIVATION_TANH > 0
    y[idx] = c_tanh(s * (dtype)0.6666) * (dtype)1.7159;
    #elif ACTIVATION_RELU > 0
    y[idx] = c_relu(s);
    #else
    #error "Activation function should be defined"
    #endif
  }
}
