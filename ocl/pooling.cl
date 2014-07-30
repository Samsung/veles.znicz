#include "defines.cl"
#include "highlight.cl"
#include "random.cl"


// Pool only full-size kernels
#define LAST_APPLY_X (SX - KX)
#define LAST_APPLY_Y (SY - KY)
#define OUT_SX (LAST_APPLY_X / SLIDE_X + 1)
#define OUT_SY (LAST_APPLY_Y / SLIDE_Y + 1)

#define TARGET_PIXEL_X (target_x / N_CHANNELS)
#define TARGET_CHANNEL (target_x % N_CHANNELS)


/// @brief Does max pooling over convolutional layer output.
/// @param h batch of input multichannel interleaved images.
/// @param y batch of output multichannel interleaved images.
/// @param h_offs indexes of y value in corresponding to it h array.
/// @details If ABS_VALUES is defined, compare absolute values; otherwise,
/// as usual.
/// Should be defined externally:
///          SX - input image width,
///          SY - input image height,
///          N_CHANNELS - number of input channels,
///          KX - pooling kernel width,
///          KY - pooling kernel height,
///          SLIDE_X - kernel sliding by x-axis,
///          SLIDE_Y - kernel sliding by y-axis.
///          Kernel should be run as:
///          global_size = [out_width, out_height],
///          local_size = None.
__kernel
void do_max_pooling(__global const dtype    /* IN */    *h,
                    __global dtype         /* OUT */    *y,
                    __global int           /* OUT */    *h_offs) {
  dtype max_vle = -MAXFLOAT;
#ifdef ABS_VALUES
  dtype max_absvle = -1;
#endif
  int max_offs = 0;
  int target_x = get_global_id(0),
      target_y = get_global_id(1);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      dtype vle = h[offs + x];
#ifdef ABS_VALUES
      dtype absvle = fabs(vle);
      bool hit = (absvle > max_absvle);
      max_absvle = (hit) ? absvle : max_absvle;
#else
      bool hit = (vle > max_vle);
#endif
      max_vle = (hit) ? vle : max_vle;
      max_offs = (hit) ? offs + x : max_offs;
    }
  }

  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  y[idx] = max_vle;
  h_offs[idx] = max_offs;
}


/// @brief Does avg pooling over convolutional layer output.
/// @param h batch of input multichannel interleaved images.
/// @param y batch of output multichannel interleaved images.
__kernel
void do_avg_pooling(__global const dtype    /* IN */    *h,
                    __global dtype         /* OUT */    *y) {

  dtype smm = 0;
  int target_x = get_global_id(0),
      target_y = get_global_id(1);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;

  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      smm += h[offs + x];
    }
  }

  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  y[idx] = smm / (KX * KY);
}


/// @brief Does stochastic pooling over convolutional layer output.
/// @param h batch of input multichannel interleaved images.
/// @param y batch of output multichannel interleaved images.
/// @param h_offs indexes of y value in corresponding to it h array.
/// @param rand random numbers.
/// @details If ABS_VALUES is defined, use absolute values; otherwise,
/// discard negative ones.
#if KX * KY > 65536
#error "Too large kernel size for the current stochastic pooling implementation"
#endif
__kernel
void do_stochastic_pooling(__global const dtype    /* IN */    *h,
                           __global dtype         /* OUT */    *y,
                           __global int           /* OUT */    *h_offs,
                           __global ushort    /* IN, OUT */    *rand) {
  int target_x = get_global_id(0),
      target_y = get_global_id(1);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;
  int original_offset = offs;
  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  dtype sum = 0;

  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      dtype val = h[offs + x];
#ifdef ABS_VALUES
      val = fabs(val);
#else
      val = max(val, (dtype)0);
#endif
      sum += val;
    }
  }

  ushort random = rand[idx];
  // The index of the passed through
  int lucky = 0;
  // All elements can be <= 0
  dtype pos_add = (sum == 0) ? 1 : 0;
  dtype pos_factor = (sum == 0) ? KX * KY : sum;
  dtype pos = (pos_factor * random) / 65536;
  sum = 0;

  // This is not just copy-paste of previous for-s
  offs = original_offset;
  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      dtype val = h[offs + x];
#ifdef ABS_VALUES
      val = fabs(val);
#else
      val = max(val, (dtype)0);
#endif
      sum += val;
      sum += pos_add;

      lucky = (pos <= sum) ? offs + x : lucky;
      sum = (pos <= sum) ? -MAXFLOAT : sum;
    }
  }

  y[idx] = h[lucky];
  h_offs[idx] = lucky;
}


#ifdef USE_POOLING_DEPOOLING
#if (KX != SLIDE_X) || (KY != SLIDE_Y)
#error "Sliding should be equal to the kernel size for the current implementation"
#endif
__kernel
void do_stochastic_pooling_depooling(__global dtype     /* IN, OUT */    *h,
                                     __global ushort    /* IN, OUT */    *rand) {
  int target_x = get_global_id(0),
      target_y = get_global_id(1);

  int start_x = TARGET_PIXEL_X * SLIDE_X * N_CHANNELS + TARGET_CHANNEL,
      start_y = target_y % OUT_SY * SLIDE_Y;
  int offs = ((target_y / OUT_SY) * SY + start_y) * SX * N_CHANNELS;
  int original_offset = offs;
  int idx = target_y * OUT_SX * N_CHANNELS + target_x;
  dtype sum = 0;

  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      dtype val = h[offs + x];
#ifdef ABS_VALUES
      val = fabs(val);
#else
      val = max(val, (dtype)0);
#endif
      sum += val;
    }
  }

  ushort random = rand[idx];
  // The index of the passed through
  int lucky = 0;
  // All elements can be <= 0
  dtype pos_add = (sum == 0) ? 1 : 0;
  dtype pos_factor = (sum == 0) ? KX * KY : sum;
  dtype pos = (pos_factor * random) / 65536;
  sum = 0;

  // This is not just copy-paste of previous for-s
  offs = original_offset;
  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      dtype val = h[offs + x];
#ifdef ABS_VALUES
      val = fabs(val);
#else
      val = max(val, (dtype)0);
#endif
      sum += val;
      sum += pos_add;

      lucky = (pos <= sum) ? offs + x : lucky;
      sum = (pos <= sum) ? -MAXFLOAT : sum;
    }
  }

  dtype chosen_value = h[lucky];

  // This is not just copy-paste of previous for-s
  offs = original_offset;
  // No partial windows at the bottom
  for (int i = 0; i < KY; i++, offs += SX * N_CHANNELS) {
    // No partial windows at the right
    for (int j = 0, x = start_x; j < KX; j++, x += N_CHANNELS) {
      h[offs + x] = (offs + x == lucky) ? chosen_value : 0;
    }
  }
}
#endif  // USE_POOLING_DEPOOLING


#undef TARGET_CHANNEL
#undef TARGET_PIXEL_X
#undef OUT_SY
#undef OUT_SX


#undef MINIMUM
