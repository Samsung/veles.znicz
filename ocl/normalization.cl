//#pragma OPENCL EXTENSION cl_khr_fp64: enable

// Local response normalization kernels for ReLU units.
// Detailed description given in article by Krizhevsky, Sutskever and Hinton:
// "ImageNet Classification with Deep Convolutional Neural Networks"
//

#include "defines.cl"
#include "highlight.cl"

//#define ALPHA 0.0001
//#define BETA 0.75
//#define K 2
//#define N 3
//#define NUM_OF_CHANS 5


void calculate_subsums(const c_dtype* h, c_dtype* subsums)
{
  for (int i = 0; i < NUM_OF_CHANS; i++) {
    subsums[i] = 0;
  }

  int min_index = 0;
  int max_index = min(N / 2, NUM_OF_CHANS - 1);
  for(int i = min_index; i <= max_index; i++) {
    subsums[0] += pow(h[i], 2);
  }

  for (int i = 1; i < NUM_OF_CHANS; i++) {
    int new_min_index = max(0, i - N / 2);
    int new_max_index = min(i + N / 2, NUM_OF_CHANS - 1);
    c_dtype subsum = subsums[i - 1];

    for(int j = min_index; j < new_min_index; j++) {
      subsum -= pow(h[j], 2);
    }

    for(int j = max_index + 1; j <= new_max_index; j++) {
      subsum += pow(h[j], 2);
    }

    subsums[i] = subsum;
    min_index = new_min_index;
    max_index = new_max_index;
  }
}

__kernel void forward(__global const c_dtype* in_data, __global c_dtype* out_data)
{
  int global_index = get_global_id(0);
  int global_offset = global_index * NUM_OF_CHANS;


  c_dtype h[NUM_OF_CHANS];
  for(int i = 0; i < NUM_OF_CHANS; i++) {
    h[i] = in_data[global_offset + i];
  }

  c_dtype subsums[NUM_OF_CHANS];
  calculate_subsums(h, subsums);

  for(int i = 0; i < NUM_OF_CHANS; i++) {
    out_data[global_offset + i] = in_data[global_offset + i] *
        pow(K + ALPHA * subsums[i], -BETA);
  }
}

__kernel void backward(__global const c_dtype* in_err_y, __global const c_dtype* in_h,
                       __global c_dtype* out_err_h)
{
  int global_index = get_global_id(0);
  int global_offset = global_index * NUM_OF_CHANS;


  c_dtype h[NUM_OF_CHANS];
  for(int i = 0; i < NUM_OF_CHANS; i++) {
    h[i] = in_h[global_offset + i];
  }

  c_dtype subsums[NUM_OF_CHANS];
  calculate_subsums(h, subsums);

  for(int i = 0; i < NUM_OF_CHANS; i++) {
    subsums[i] = K + ALPHA * subsums[i];
  }

  c_dtype local_err_y[NUM_OF_CHANS];
  for(int i = 0; i < NUM_OF_CHANS; i++) {
    local_err_y[i] = in_err_y[global_offset + i];
  }

  for(int i = 0; i < NUM_OF_CHANS; i++) {
    c_dtype delta_h = 0;

    int min_index = max(0, i - N / 2);
    int max_index = min(i + N / 2, NUM_OF_CHANS - 1);

    for(int j = min_index; j <= max_index; j++) {
      if(i == j) {
        delta_h += subsums[j];
      }
      delta_h -= 2 * ALPHA * BETA * h[i] * h[j];
      delta_h *= local_err_y[j] / pow(subsums[j], BETA + 1);
    }
    out_err_h[global_offset + i] = delta_h;
  }
}

