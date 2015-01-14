#include "gradient_descent_common.cl"

#if USE_ORTHO > 0
#include "weights_ortho.cl"
#endif

#if INCLUDE_BIAS != 0
#error "INCLUDE_BIAS should be 0"
#endif

#if (!(USE_HITS > 0)) && ((KX % SLIDE_X != 0) || (KY % SLIDE_Y != 0))
#error "Incorrect SLIDE"
#endif


#include "conv_common.cl"


/// @brief Calculate gradient for weights update.
/// @details See gradient_descent.cl.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global const dtype    /* IN */    *err_output,
                    __global const dtype    /* IN */    *input,
                    __global dtype     /* IN, OUT */    *weights,
                    __global dtype     /* IN, OUT */    *gradient,
                    __global dtype     /* IN, OUT */    *accumulated_gradient,
                    __global dtype     /* IN, OUT */    *gradient_with_moment,
                    const dtype             /* IN */    lr,
                    const dtype             /* IN */    factor_l12,
                    const dtype             /* IN */    l1_vs_l2,
                    const dtype             /* IN */    moment
#if USE_ORTHO > 0
                    , const dtype           /* IN */    factor_ortho,
                    __global const dtype    /* IN */    *col_sums
#endif
                    ) {
  #if WEIGHTS_TRANSPOSED > 0

  #define A_WIDTH ELEMENTS_PER_KERNEL
  #define B_WIDTH N_KERNELS
  #define A err_output
  #define B input

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS
  #define A_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #else

  #define A_WIDTH N_KERNELS
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define A input
  #define B err_output

  #define in_offs b_offs
  #define B_REAL_OFFS IN_REAL_OFFS
  #define B_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #endif

  #define AB_COMMON (BATCH * KERNELS_PER_SAMPLE)

  #define A_COL
  #define B_COL

  #define STORE_OUTPUT "weights_update.store_output.cl"
  #include "matrix_multiplication.cl"

  #undef A_COL
  #undef B_COL

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON
  #undef in_offs

  #undef A
  #undef B
}


#if USE_HITS > 0
__kernel
void err_output_update(__global dtype    /* IN, OUT */    *err_output,
                       __global const int     /* IN */    *hits) {
  int idx = get_global_id(0);
  int n = hits[idx];
  err_output[idx] /= n ? n : 1;
}
#else
__kernel
void err_output_update(__global dtype /* IN, OUT */ *err_output) {
  err_output[get_global_id(0)] /= (KX / SLIDE_X) * (KY / SLIDE_Y);
}
#endif
