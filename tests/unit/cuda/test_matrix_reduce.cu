#include "defines.cu"

extern "C"
__global__ void test(const dtype *A, dtype *b) {
  #include "matrix_reduce.cu"
  if (!tx) {
    sum += AS[0];
    b[bx] = sum;
  }
}
