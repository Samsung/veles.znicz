/*! @file units.h
 *  @brief The forward declarations of all Znicz units.
 *  @author Vadim Markovtsev <v.markovtsev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_ZNICZ_UNITS_H_
#define INC_ZNICZ_UNITS_H_

#include <veles/unit_factory.h>

#if __GNUC__ >= 4
#pragma GCC visibility push(default)
#endif

namespace veles {
namespace znicz {

class All2AllTanh;
class All2AllSoftmax;
class All2AllLinear;

DECLARE_UNIT(All2AllLinear);
DECLARE_UNIT(All2AllSoftmax);
DECLARE_UNIT(All2AllTanh);

}  // namespace znicz
}  // namespace veles

#if __GNUC__ >= 4
#pragma GCC visibility pop
#endif

#endif  // INC_ZNICZ_UNITS_H_
