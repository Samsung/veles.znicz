#ifndef _POOLING_COMMON_
#define _POOLING_COMMON_

#include "defines.cu"
#include "highlight.cuh"

#if (KX < SLIDE_X) || (KY < SLIDE_Y)
#error "Sliding should not be greater than kernel size"
#endif

#define LAST_X (SX - KX)
#define LAST_Y (SY - KY)

#if LAST_X % SLIDE_X == 0
#define OUT_SX (LAST_X / SLIDE_X + 1)
#else
#define OUT_SX (LAST_X / SLIDE_X + 2)
#endif
#if LAST_Y % SLIDE_Y == 0
#define OUT_SY (LAST_Y / SLIDE_Y + 1)
#else
#define OUT_SY (LAST_Y / SLIDE_Y + 2)
#endif

#define TARGET_PIXEL_X (target_x / N_CHANNELS)
#define TARGET_CHANNEL (target_x % N_CHANNELS)

#endif  // _POOLING_COMMON_
