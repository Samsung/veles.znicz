#include "defines.cl"

/// @brief Joins two minibatch inputs into one.
/// @param output Output.
/// @param a First input.
/// @param b Second input.
/// @param a_size Size of a's sample.
/// @param b_size Size of b's sample.
/// @param output_sample_size Size of an output sample.
/// @param in_sample_output_offs Offset in the output sample where to put the result.
/// @details etype is an element type and should be defined externally;
///          workgroup dimension 0 corresponds to in-sample offset,
///          workgroup dimension 1 corresponds to the sample index.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
__kernel
void join2(__global etype *output, __global etype *a, __global etype *b,
           const int a_size, const int b_size,
           const int in_sample_output_offs,
           const int output_sample_size) {
  int offs = get_global_id(0);
  int i_sample = get_global_id(1);
  output[i_sample * output_sample_size + in_sample_output_offs + offs] =
    offs < a_size ? a[i_sample * a_size + offs] : b[i_sample * b_size + offs - a_size];
}
