#ifndef _GRADIENT_DESCENT_COMMON_
#define _GRADIENT_DESCENT_COMMON_

#include "defines.cu"

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

#ifndef ACCUMULATE_GRADIENT
#error "ACCUMULATE_GRADIENT should be defined"
#endif
#define OP_NONE 0
#define OP_STORE 1
#define OP_ADD 2
#define OP_FLUSH 3
#if (ACCUMULATE_GRADIENT >= 0) && (ACCUMULATE_GRADIENT <= 3)
// All Ok
#else
#error "Incorrect ACCUMULATE_GRADIENT"
#endif

#ifndef APPLY_GRADIENT
#error "APPLY_GRADIENT should be defined"
#endif


#define gradient_step_l12(weight, factor, l1_vs_l2) (factor * (((dtype)1.0 - l1_vs_l2) * weight + (dtype)0.5 * l1_vs_l2 * SIGN(weight)))


#endif  // _GRADIENT_DESCENT_COMMON_
