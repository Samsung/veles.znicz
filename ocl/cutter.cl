#include "defines.cl"
#include "highlight.cl"

KERNEL_CLEAR(clear_err_input, dtype)

__kernel void cutter_1d_forward(__global const dtype     *input,
								const int				 input_start,
                                const dtype              alpha,
                                const int                input_sample_size,
                                __global dtype           *output,
								const int				 output_start,
                                const dtype              beta,
								const int                output_sample_size) {
  input += input_start;
  output += output_start;
  int sample_number = get_global_id(0);
  int sample_offset = get_global_id(1);
  dtype x = input[sample_number * input_sample_size + sample_offset] * alpha;
  int output_offset = sample_number * output_sample_size + sample_offset;
  output[output_offset] = beta ? output[output_offset] * beta + x : x;
}
