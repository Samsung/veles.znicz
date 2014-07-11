#ifndef _CONV_COMMON_
#define _CONV_COMMON_

/// @brief Common defines for convolutional kernels.
/// @details Should be defined externally:
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

#define SX_FULL (SX + PAD_LEFT + PAD_RIGHT)
#define SY_FULL (SY + PAD_TOP + PAD_BOTTOM)

#define KERNEL_APPLIES_PER_WIDTH ((SX_FULL - KX) / SLIDE_X + 1)
#define KERNEL_APPLIES_PER_HEIGHT ((SY_FULL - KY) / SLIDE_Y + 1)
#define KERNELS_PER_SAMPLE (KERNEL_APPLIES_PER_WIDTH * KERNEL_APPLIES_PER_HEIGHT)

#define ELEMENTS_PER_KERNEL (N_CHANNELS * KX * KY)
#define KERNEL_APPLY_NUMBER (in_offs / ELEMENTS_PER_KERNEL)

#define OFFS_IN_KERNEL (in_offs % ELEMENTS_PER_KERNEL)
#define PLAIN_ROW_IN_KERNEL (OFFS_IN_KERNEL / (N_CHANNELS * KX))
#define PLAIN_COL_CHANNEL_IN_KERNEL (OFFS_IN_KERNEL % (N_CHANNELS * KX))

#define SAMPLE_NUMBER (KERNEL_APPLY_NUMBER / KERNELS_PER_SAMPLE)
#define KERNEL_APPLY_IN_SAMPLE (KERNEL_APPLY_NUMBER % KERNELS_PER_SAMPLE)
#define VIRT_ROW_IN_SAMPLE (KERNEL_APPLY_IN_SAMPLE / KERNEL_APPLIES_PER_WIDTH)
#define VIRT_COL_IN_SAMPLE (KERNEL_APPLY_IN_SAMPLE % KERNEL_APPLIES_PER_WIDTH)

#define SAMPLE_ROW (VIRT_ROW_IN_SAMPLE * SLIDE_Y + PLAIN_ROW_IN_KERNEL - PAD_TOP)
#define SAMPLE_COL_CHANNEL (VIRT_COL_IN_SAMPLE * SLIDE_X * N_CHANNELS + PLAIN_COL_CHANNEL_IN_KERNEL - PAD_LEFT * N_CHANNELS)

#define IN_REAL_OFFS_VALID ((SAMPLE_ROW >= 0) && (SAMPLE_ROW < SY) && (SAMPLE_COL_CHANNEL >= 0) && (SAMPLE_COL_CHANNEL < SX * N_CHANNELS))
#define IN_REAL_OFFS ((SAMPLE_NUMBER * SY + SAMPLE_ROW) * (SX * N_CHANNELS) + SAMPLE_COL_CHANNEL)

#endif  // _CONV_COMMON_
