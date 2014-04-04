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
///          N_KERNELS - number of kernels (i.e. neurons),
///          PAD_TOP - padding-top,
///          PAD_LEFT - padding-left,
///          PAD_BOTTOM - padding-bottom,
///          PAD_RIGHT - padding-right,
///          SLIDE_X - kernel sliding by x-axis,
///          SLIDE_Y - kernel sliding by y-axis.
///          The process we are doing here equal to the convolution with reflected weights matrix,
///          and it is so in favor of the nature of the process of applying neurons (i.e. kernels) to each point of the input image.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void feed_layer(__global c_dtype /*IN*/ *h, __global c_dtype /*IN*/ *weights,
                __global c_dtype /*OUT*/ *y, __global c_dtype /*IN*/ *bias) {

  #define SX_FULL (SX + PAD_LEFT + PAD_RIGHT)
  #define SY_FULL (SY + PAD_TOP + PAD_BOTTOM)

  #define KERNEL_APPLIES_PER_WIDTH ((SX_FULL - KX) / SLIDE_X + 1)
  #define KERNEL_APPLIES_PER_HEIGHT ((SY_FULL - KY) / SLIDE_Y + 1)
  #define KERNELS_PER_SAMPLE (KERNEL_APPLIES_PER_WIDTH * KERNEL_APPLIES_PER_HEIGHT)

  #define A_WIDTH (BATCH * KERNELS_PER_SAMPLE)
  #define B_WIDTH N_KERNELS
  #define AB_COMMON (KX * KY * N_CHANNELS)

  #define A h
  #define B weights
  #define C y

  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #define ELEMENTS_PER_KERNEL (N_CHANNELS * KX * KY)
  #define KERNEL_APPLY_NUMBER (a_offs / ELEMENTS_PER_KERNEL)
  #define OFFS_IN_KERNEL (a_offs % ELEMENTS_PER_KERNEL)
  #define PLAIN_ROW_IN_KERNEL (OFFS_IN_KERNEL / (N_CHANNELS * KX))
  #define PLAIN_COL_IN_KERNEL (OFFS_IN_KERNEL % (N_CHANNELS * KX))
  #define SAMPLE_NUMBER (KERNEL_APPLY_NUMBER / KERNELS_PER_SAMPLE)
  #define KERNEL_APPLY_IN_SAMPLE (KERNEL_APPLY_NUMBER % KERNELS_PER_SAMPLE)
  #define VIRT_ROW_IN_SAMPLE (KERNEL_APPLY_IN_SAMPLE / KERNEL_APPLIES_PER_WIDTH)
  #define VIRT_COL_IN_SAMPLE (KERNEL_APPLY_IN_SAMPLE % KERNEL_APPLIES_PER_WIDTH)

  #define SAMPLE_ROW (VIRT_ROW_IN_SAMPLE * SLIDE_Y + PLAIN_ROW_IN_KERNEL - PAD_TOP)
  #define SAMPLE_COL (VIRT_COL_IN_SAMPLE * SLIDE_X * N_CHANNELS + PLAIN_COL_IN_KERNEL - PAD_LEFT * N_CHANNELS)

  #define A_REAL_OFFS_VALID ((SAMPLE_ROW >= 0) && (SAMPLE_ROW < SY) && (SAMPLE_COL >= 0) && (SAMPLE_COL < SX * N_CHANNELS))
  #define A_REAL_OFFS ((SAMPLE_NUMBER * SY + SAMPLE_ROW) * (SX * N_CHANNELS) + SAMPLE_COL)

  #include "matrix_multiplication.cl"

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
