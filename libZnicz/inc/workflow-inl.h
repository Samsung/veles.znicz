/*! @file workflow-inl.h
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

#ifndef WORKFLOW_INL_H_
#define WORKFLOW_INL_H_

#include <memory>
#include <algorithm>
#include <simd/inc/simd/memory.h>

namespace Veles {
namespace Znicz {

/** @brief Execute the workflow
 *  @param begin Iterator to the first element of initial data
 *  @param end Iterator to the end of initial data
 *  @param out Output iterator for the result
 */
template<class InputIterator, class OutputIterator>
void Workflow::Execute(InputIterator begin, InputIterator end, OutputIterator out) const {
  size_t max_size = get_max_unit_size();
  std::unique_ptr<float> input(mallocf(max_size), std::free);
  std::unique_ptr<float> output(mallocf(max_size), std::free);
  std::copy(begin, end, input.get());

  float* curr_in = input.get();
  float* curr_out = output.get();

  if(!units_.empty()) {
    for(const auto& unit : units_) {
      unit->Execute(curr_in, curr_out);
      std::swap(curr_in, curr_out);
    }
    std::copy(curr_in, curr_in + units_.back()->outputs(), out);
  }
}

}  // namespace Znicz
}  // namespace Veles

#endif  // WORKFLOW_INL_H_
