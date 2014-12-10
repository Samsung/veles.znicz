#include "defines.cl"

__kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
void test(__global dtype *A, __global dtype *b) {
  #include "matrix_reduce.cl"
  if (!tx) {
    sum += AS[0];
    b[bx] = sum;
  }
}
