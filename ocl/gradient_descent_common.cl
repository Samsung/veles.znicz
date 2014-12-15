#ifndef _GRADIENT_DESCENT_COMMON_
#define _GRADIENT_DESCENT_COMMON_

#include "defines.cl"
#include "highlight.cl"

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

#ifndef STORE_GRADIENT
#error "STORE_GRADIENT should be defined"
#endif
#define OP_DEFAULT 0
#define OP_STORE 1
#define OP_ACCUMULATE 2
#define OP_FLUSH 3
#if (STORE_GRADIENT >= 0) && (STORE_GRADIENT <= 3)
// All Ok
#else
#error "Incorrect STORE_GRADIENT"
#endif

#ifndef APPLY_GRADIENT
#error "APPLY_GRADIENT should be defined"
#endif


#define gradient_step_l12(weight, factor, l1_vs_l2) (factor * (((dtype)1.0 - l1_vs_l2) * weight + (dtype)0.5 * l1_vs_l2 * sign(weight)))


#endif  // _GRADIENT_DESCENT_COMMON_
