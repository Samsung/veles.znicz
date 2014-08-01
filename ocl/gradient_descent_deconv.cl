#include "gradient_descent_common.cl"

#if USE_ORTHO > 0
#include "weights_ortho.cl"
#endif

#if INCLUDE_BIAS != 0
#error "INCLUDE_BIAS should be 0"
#endif

#if (KX % SLIDE_X != 0) || (KY % SLIDE_Y != 0)
#error "Incorrect SLIDE"
#endif

#include "conv.cl"

#if (STORE_GRADIENT > 0) || (APPLY_GRADIENT > 0)
/// @brief Calculate gradient for weights update.
/// @details See gradient_descent.cl.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void weights_update(__global const dtype    /* IN */    *err_y,
                    __global const dtype    /* IN */    *h,
                    __global dtype     /* IN, OUT */    *weights,
                    __global dtype     /* IN, OUT */    *gradient,
                    const dtype             /* IN */    lr,
                    const dtype             /* IN */    factor_l12,
                    const dtype             /* IN */    l1_vs_l2,
                    const dtype             /* IN */    gradient_moment
#if USE_ORTHO > 0
                    , const dtype           /* IN */    factor_ortho,
                    __global const dtype    /* IN */    *col_sums
#endif
                    ) {
  #if WEIGHTS_TRANSPOSED > 0

  #define A_WIDTH ELEMENTS_PER_KERNEL
  #define B_WIDTH N_KERNELS
  #define A err_y
  #define B h

  #define in_offs a_offs
  #define A_REAL_OFFS IN_REAL_OFFS
  #define A_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #else

  #define A_WIDTH N_KERNELS
  #define B_WIDTH ELEMENTS_PER_KERNEL
  #define A h
  #define B err_y

  #define in_offs b_offs
  #define B_REAL_OFFS IN_REAL_OFFS
  #define B_REAL_OFFS_VALID IN_REAL_OFFS_VALID

  #endif

  #define AB_COMMON (BATCH * KERNELS_PER_SAMPLE)

  #define A_COL
  #define B_COL

  #include "matrix_multiplication.cl"

  #undef A_COL
  #undef B_COL

  #undef A_WIDTH
  #undef B_WIDTH
  #undef AB_COMMON
  #undef in_offs

  #undef A
  #undef B

  if (valid) {
    dtype weight = weights[idx];
    dtype gd = -lr * (sum + gradient_step_l12(weight, factor_l12, l1_vs_l2)
#if USE_ORTHO > 0
    #if WEIGHTS_TRANSPOSED > 0
               + gradient_step_ortho(weight, factor_ortho, get_global_id(1), Y, col_sums)
    #else
               + gradient_step_ortho(weight, factor_ortho, get_global_id(0), Y, col_sums)
    #endif
#endif
               );
    #if STORE_GRADIENT > 0
    gd += gradient[idx] * gradient_moment;
    gradient[idx] = gd;
    #endif
    #if APPLY_GRADIENT > 0
    weights[idx] = weight + gd;
    #endif
  }
}
#endif

__kernel
void err_output_update(__global dtype /* IN, OUT */ *err_output) {
  err_output[get_global_id(0)] /= (KX / SLIDE_X) * (KY / SLIDE_Y);
}
