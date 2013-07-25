/*! @file workflow.cc
 *  @brief VELES neural network workflow
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include <numeric>
#include <algorithm>
#include "inc/workflow.h"

namespace Veles {
namespace Znicz {

/** @brief Load %Workflow data from string
 *  @param data %Workflow declaration in VELES format
 *  @TODO Define VELES workflow loading
 */
void Workflow::Load(const std::string& data) {

}

/** @brief Get maximum input and output size of containing units
 *  @return Maximum size
 */
size_t Workflow::get_max_unit_size() const noexcept {
  auto max_func = [](size_t curr, const std::unique_ptr<Unit>& other) -> size_t
      { return std::max(curr, other->outputs()); };
  return std::max(
      std::accumulate(units_.begin(), units_.end(), static_cast<size_t>(0), max_func),
      !units_.empty() ? units_.front()->inputs() : 0);
}

}  // namespace Znicz
}  // namespace Veles

