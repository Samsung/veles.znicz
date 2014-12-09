#include "defines.cu"
#include "highlight.cuh"

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
__global__ void join2(etype *output, etype *a, etype *b, const int a_size,
                      const int b_size, const int in_sample_output_offs,
                      const int output_sample_size) {
  int offs = threadIdx.x + blockIdx.x * blockDim.x;
  int i_sample = threadIdx.y + blockIdx.y * blockDim.y;
  output[i_sample * output_sample_size + in_sample_output_offs + offs] =
      offs < a_size ?
          a[i_sample * a_size + offs] : b[i_sample * b_size + offs - a_size];
}
