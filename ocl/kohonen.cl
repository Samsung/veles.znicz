#include "defines.cl"
#include "highlight.cl"

#ifndef __OPENCL_VERSION__
#define FORWARD
#define TRAIN
#endif


/// @brief Calculates distances between input and neuron weights.
/// @param input The input samples each of length SAMPLE_LENGTH.
/// @param weights The neurons' weights.
/// @param output Distances(input, weights).
/// @details Must be defined externally:
///          BATCH - minibatch size,
///          SAMPLE_LENGTH - the length of each sample,
///          NEURONS_NUMBER - the number of neurons.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void calculate_distances(__global const dtype    /* IN */    *input,
                         __global const dtype    /* IN */    *weights,
                         __global dtype         /* OUT */    *output) {
  #define A_WIDTH BATCH
  #define B_WIDTH NEURONS_NUMBER
  #define AB_COMMON SAMPLE_LENGTH

  #define A input
  #define B weights

  #ifdef WEIGHTS_TRANSPOSED
  #define B_COL
  #endif

  #define MULTIPLY(a, b) (((a) - (b)) * ((a) - (b)))

  #include "matrix_multiplication.cl"

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON

  #undef A
  #undef B

  if (valid) {
    output[idx] = sum;
  }
}

/// @brief Calculates the winning neuron indices for each sample.
/// @param dists Values to find minimum of.
/// @param argmin Indices of min elements. May be not initialized.
/// @details Must be defined externally:
///          BATCH - the number of samples, the size of argmin,
///          CHUNK_SIZE - the number of distances processed by each thread,
///          NEURONS_NUMBER - the number of neurons.
#if NEURONS_NUMBER % CHUNK_SIZE > 0
#define WORK_GROUP_SIZE (NEURONS_NUMBER / CHUNK_SIZE + 1)
#else
#define WORK_GROUP_SIZE (NEURONS_NUMBER / CHUNK_SIZE)
#endif
__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
void calculate_argmin(__global const dtype /* IN */   *dists,
                      __global int         /* OUT */  *argmin,
                      __global volatile int/* OUT */  *winners) {

  int tx = get_local_id(0); // from 0 to WORK_GROUP_SIZE - 1

  __local dtype mins[BATCH * WORK_GROUP_SIZE];
  __local dtype argmins[BATCH * WORK_GROUP_SIZE];

  for (int sample = 0; sample < BATCH; sample++) {
    dtype min_value = MAXFLOAT;
    int min_index = -1;
    int offset = sample * NEURONS_NUMBER;
    for (int i = offset + tx * CHUNK_SIZE;
         i < offset + MIN((tx + 1) * CHUNK_SIZE, NEURONS_NUMBER);
         i++) {
      dtype value = dists[i];
      if (value < min_value) {
        min_value = value;
        min_index = i - offset;
      }
    }
    mins[sample * WORK_GROUP_SIZE + tx] = min_value;
    argmins[sample * WORK_GROUP_SIZE + tx] = min_index;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#if BATCH > WORK_GROUP_SIZE
  int max_sample_add = 0;
  if (tx == WORK_GROUP_SIZE - 1) {
    max_sample_add = BATCH - (BATCH / WORK_GROUP_SIZE) * WORK_GROUP_SIZE;
  }
  for (int sample = (BATCH / WORK_GROUP_SIZE) * tx;
      sample < (BATCH / WORK_GROUP_SIZE) * (tx + 1) + max_sample_add;
      sample++) {
    int offset = sample * WORK_GROUP_SIZE;
    dtype min_value = MAXFLOAT;
    int min_index = -1;
    for (int i = offset; i < offset + WORK_GROUP_SIZE; i++) {
      dtype value = mins[i];
      if (value < min_value) {
        min_value = value;
        min_index = argmins[i];
      }
    }
    argmin[sample] = min_index;
    if (winners) {
      atomic_inc(winners + min_index);
    }
  }
#else
  if (tx < BATCH) {
    int offset = tx * WORK_GROUP_SIZE;
    dtype min_value = MAXFLOAT;
    int min_index = -1;
    for (int i = offset; i < offset + WORK_GROUP_SIZE; i++) {
      dtype value = mins[i];
      if (value < min_value) {
        min_value = value;
        min_index = argmins[i];
      }
    }
    argmin[tx] = min_index;
    if (winners) {
      atomic_inc(winners + min_index);
    }
  }
#endif
}
#undef WORK_GROUP_SIZE

#ifdef FORWARD
/// @brief Records winners of the absolutely indexed samples.
/// @param argmins Winning neuron indices in current minibatch.
/// @param batch_offset The sample index offset.
/// @param total The array of winning neurons, absolutely indexed.
/// @details Must be defined externally:
///          BATCH - the number of input samples,
///          COPY_CHUNK_SIZE - the number of winners processed at a time.
__kernel
void set_total(__global const int    /* IN */    *argmins,
               const int             /* IN */    batch_offset,
               __global int         /* OUT */    *total) {
  int offset = get_global_id(0);
  for (int i = offset * COPY_CHUNK_SIZE;
      i < MIN((offset + 1) * COPY_CHUNK_SIZE, BATCH);
      i++) {
    total[i + batch_offset] = argmins[i];
  }
}
#endif

#ifdef TRAIN
/// @brief Computes gravity function from argmin neuron to all others.
/// @param argmin Indexes of neurons with min distances to inputs.
/// @param coords Neuron coordinates in Euclidian space.
/// @param gravity Output gravity.
/// @param sigma Effective radius.
/// @details Must be defined externally:
///          NEURONS_NUMBER - output size,
///          coord_type - type for coordinates of neuron in space (float2).
__kernel
void compute_gravity(__global const int           /* IN */    *argmin,
                     __global const coord_type    /* IN */    *coords,
                     const dtype                  /* IN */    sigma,
                     __global dtype              /* OUT */    *gravity) {
  int src = get_global_id(0);
  int dst = get_global_id(1);
  dtype d = distance(coords[argmin[src]], coords[dst]);
  gravity[src * NEURONS_NUMBER + dst] = exp((d * d) / (-2 * sigma * sigma));
}

/// @brief Updates weights according to Kohonen's learning algorithm.
/// @param input The input samples.
/// @param weights The Weights.
/// @param gravity Gravity function for each neuron relative to the winner.
/// @param time_reciprocal 1 / t
/// @details Must be defined externally:
///          BATCH - the number of samples, the size of argmin,
///          SAMPLE_LENGTH - the number of weights in each neuron,
///          GRADIENT_CHUNK_SIZE - the number of weights processed by each thread,
///          NEURONS_NUMBER - the number of neurons.
__kernel
void apply_gradient(__global const dtype    /* IN */    *input,
                    __global const dtype    /* IN */    *gravity,
                    const dtype             /* IN */    time_reciprocal,
                    __global dtype     /* IN, OUT */    *weights) {
  int chunk_number = get_global_id(0);
  int tindex = get_global_id(1);
  int weight_number = chunk_number * GRADIENT_CHUNK_SIZE + tindex;
  if (weight_number >= SAMPLE_LENGTH) {
    return;
  }

  dtype orig_weights[NEURONS_NUMBER];
  for (int n = 0; n < NEURONS_NUMBER; n++) {
#ifndef WEIGHTS_TRANSPOSED
    int twi = n * SAMPLE_LENGTH + weight_number;
#else
    int twi = n + weight_number * NEURONS_NUMBER;
#endif
    orig_weights[n] = weights[twi];
  }

  for (int sample = 0; sample < BATCH; sample++) {
    for (int n = 0; n < NEURONS_NUMBER; n++) {
#ifndef WEIGHTS_TRANSPOSED
      int twi = n * SAMPLE_LENGTH + weight_number;
#else
      int twi = n + weight_number * NEURONS_NUMBER;
#endif
      weights[twi] += gravity[sample * NEURONS_NUMBER + n] *
          time_reciprocal * (input[sample * SAMPLE_LENGTH + weight_number] -
          orig_weights[n]);
    }
  }
}
#endif  // TRAIN
