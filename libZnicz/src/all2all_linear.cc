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

#include "src/all2all_linear.h"

namespace veles {
namespace znicz {

const std::string All2AllLinear::uuid_ = "58a5eadf-ae1e-498f-bf35-7d93939c4c86";

const std::string& All2AllLinear::Uuid() const noexcept {
  return uuid_;
}

REGISTER_UNIT(All2AllLinear);

}  // namespace znicz
}  // namespace veles
