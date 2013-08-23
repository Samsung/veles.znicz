/*! @file all2all_tanh.cc
 *  @brief "All to all" neural network layer with linear activation function
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */


#include <veles/unit_factory.h>
#include "src/all2all_linear.h"

namespace Veles {
namespace Znicz {

std::string All2AllLinear::Name() const noexcept {
  return "All2All";
}

REGISTER_UNIT(All2AllLinear);

}  // namespace Znicz
}  // namespace Veles
