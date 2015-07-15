#include "defines.cu"
#include "highlight.cu"


extern "C"
__global__ void cutter_1d_forward(const dtype     *input,
                                  const dtype     alpha,
                                  const int       input_sample_size,
                                  dtype           *output,
                                  const dtype     beta,
                                  const int       output_sample_size,
                                  const int       interval_length,
                                  const int       limit) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < limit) {
    int sample_number = index / interval_length;
    int sample_offset = index % interval_length;
    dtype x = input[sample_number * input_sample_size + sample_offset] * alpha;
    int output_offset = sample_number * output_sample_size + sample_offset;
    output[output_offset] = beta ? output[output_offset] * beta + x : x;
  }
}
