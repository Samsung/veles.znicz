#ifndef _GRADIENT_DESCENT_COMMON_
#define _GRADIENT_DESCENT_COMMON_

#include "defines.cl"
#include "highlight.cl"

#ifndef WEIGHTS_TRANSPOSED
#error "WEIGHTS_TRANSPOSED should be defined"
#endif

#ifndef ACCUMULATE_GRADIENT
#error "ACCUMULATE_GRADIENT should be defined"
#endif

#ifndef APPLY_GRADIENT
#error "APPLY_GRADIENT should be defined"
#endif


#define gradient_step_l12(weight, factor, l1_vs_l2) (factor * (((dtype)1.0 - l1_vs_l2) * weight + (dtype)0.5 * l1_vs_l2 * sign(weight)))


#endif  // _GRADIENT_DESCENT_COMMON_
