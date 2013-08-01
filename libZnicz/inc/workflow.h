/*! @file workflow.h
 *  @brief New file description.
 *  @author Ernesto Sanches <ernestosanches@gmail.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#ifndef INC_WORKFLOW_H_
#define INC_WORKFLOW_H_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <veles/unit.h>
#include <simd/inc/simd/memory.h>

namespace Veles {
namespace Znicz {

/** @brief VELES Neural network workflow */
class Workflow {
 public:
  /** @brief Constructs empty workflow */
  Workflow() = default;

  /** @brief Constructs workflow from data in VELES format */
  explicit Workflow(const std::string& data) {
    Load(data);
  }

  /** @brief Load Workflow data from string
   *  @param data Workflow declaration in VELES format
   */
  void Load(const std::string& data);

  /** @brief Execute the workflow
   *  @param begin Iterator to the first element of initial data
   *  @param end Iterator to the end of initial data
   *  @param out Output iterator for the result
   */
  template<class InputIterator, class OutputIterator>
  void Execute(InputIterator begin, InputIterator end,
                         OutputIterator out) const {
    size_t max_size = get_max_unit_size();
    std::unique_ptr<float> input(mallocf(max_size), std::free);
    std::unique_ptr<float> output(mallocf(max_size), std::free);
    std::copy(begin, end, input.get());

    float* curr_in = input.get();
    float* curr_out = output.get();

    if (!units_.empty()) {
      for (const auto& unit : units_) {
        unit->Execute(curr_in, curr_out);
        std::swap(curr_in, curr_out);
      }
      std::copy(curr_in, curr_in + units_.back()->OutputCount(), out);
    }
  }

 private:
  /** @brief Get maximum input and output size of containing units
   *  @return Maximum size
   */
  size_t get_max_unit_size() const noexcept;

  std::vector<std::shared_ptr<Unit>> units_;
};

}  // namespace Znicz
}  // namespace Veles

#endif  // INC_WORKFLOW_H_
