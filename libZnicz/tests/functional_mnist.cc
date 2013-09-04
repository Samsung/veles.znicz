/*! @file functional_mnist.cc
 *  @brief New file description.
 *  @author Bulychev Egor <e.bulychev@samsung.com>
 *  @version 1.0
 *
 *  @section Notes
 *  This code partially conforms to <a href="http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml">Google C++ Style Guide</a>.
 *
 *  @section Copyright
 *  Copyright 2013 Samsung R&D Institute Russia
 */

#include "src/all2all_linear.h"
#include "src/all2all_tanh.h"
#include <veles/workflow_loader.h>
#include <unistd.h>
#include <float.h>
#include <string>
#include <gtest/gtest.h>

using std::string;
using std::static_pointer_cast;

namespace Veles {

class WorkflowLoaderTest: public ::testing::Test {
 public:
  WorkflowLoaderTest() {
    current_path = "";
    char  currentPath[FILENAME_MAX];

    if (!getcwd(currentPath, sizeof(currentPath))) {
      fprintf(stderr, "Can't locate current directory\n");
    } else {
      current_path = currentPath;
    }
    string temp_path1 = current_path + "/../workflow_files/";
    string temp_path2 = current_path + "/tests/workflow_files/";
    string temp_path3 = current_path + "/workflow_files/";
    // Check existence of archive
    if (access(temp_path1.c_str(), 0) != -1) {
      current_path = temp_path1;  // "/workflow_desc_files/"
    } else if (access(temp_path2.c_str(), 0) != -1) {
      current_path = temp_path2;  // "/tests/workflow_desc_files/"
    } else if (access(temp_path3.c_str(), 0) != -1) {
      current_path = temp_path3;  // "/tests/workflow_desc_files/"
    } else {
      fprintf(stderr, "%s\n", current_path.c_str());
      current_path = "";  // Error
    }
  }

  void Mnist() {
    if (current_path == string("")) {
      FAIL();  // Can't find folder workflow_desc_files
    }
    // Check everything
    WorkflowLoader testWorkflowLoader;

    string temp = current_path + "workflow.tar.gz";
    ASSERT_NO_THROW(testWorkflowLoader.Load(temp));

    UnitDescription testUnitAll2AllTanh;
    UnitDescription testUnitAll2All;

    ASSERT_EQ(string("All2AllTanh"),
              testWorkflowLoader.workflow_desc_.Units.at(0).Name);
    ASSERT_EQ(string("All2All"),
              testWorkflowLoader.workflow_desc_.Units.at(1).Name);

    for (auto& it : testWorkflowLoader.workflow_desc_.Units) {
      if (it.Name == "All2AllTanh") {
        testUnitAll2AllTanh = it;
      } else if (it.Name == "All2All") {
        testUnitAll2All = it;
      } else {
        FAIL();
      }
    }

    Workflow testWorkflow;
    ASSERT_NO_THROW(testWorkflow = testWorkflowLoader.GetWorkflow());

    ASSERT_EQ(size_t(2), testWorkflow.Size());

    ASSERT_EQ(size_t(2), testWorkflowLoader.workflow_desc_.Units.size());
    // Check unit All2AllTanh name, that unit is exist
    ASSERT_EQ(string("All2AllTanh"), testUnitAll2AllTanh.Name);
    // Check size of All2AllTanh input
    // (60 images * 28px * 28px * float( 4 bytes) = 188160 bytes)
    ASSERT_EQ(size_t(188160/4), *std::static_pointer_cast<size_t>(
        testUnitAll2AllTanh.Properties.at("input_length")));

    ASSERT_EQ(*std::static_pointer_cast<size_t>(
        testUnitAll2AllTanh.Properties.at("output_length")),
              (*std::static_pointer_cast<size_t>(
                  testUnitAll2All.Properties.at("input_length"))));
    // Check unit All2All name, that unit is exist
    ASSERT_EQ(string("All2All"), testUnitAll2All.Name);
    // Check size of All2All output
    // (60 images * 10 floats(4 bytes)= 2400 bytes)
    ASSERT_EQ(size_t(2400/4), *std::static_pointer_cast<size_t>(
        testUnitAll2All.Properties.at("output_length")));

    fprintf(stderr,"Before Znicz::Execute.\n\n\n");
    const size_t pxNumber = 784;  // Number of pixels in image
    const size_t imgNumber = 60;  // batch contents 60 frames
    const size_t digitNumber = 10;  // Number of digits

    for (size_t i = 0; i < imgNumber; ++i) {
      auto begin = static_cast<float*>(
              testUnitAll2AllTanh.Properties.at("input").get());
      float array[pxNumber];
      for (size_t j = 0; j < pxNumber; ++j) {
        array[j] = begin[j + pxNumber*i];
      }

      auto resultBegin = static_cast<float*>(
          testUnitAll2All.Properties.at("output").get());// + i*digitNumber;
      float result[digitNumber];
      for (size_t j = 0; j < digitNumber; ++j) {
        result[j] = resultBegin[j + digitNumber*i];
      }

      float resultFromExecute[digitNumber];

      /*ASSERT_NO_FATAL_FAILURE*/
      testWorkflow.Execute(array, array + pxNumber - 1, resultFromExecute);

      if (findMaxFromNFloats(result) != findMaxFromNFloats(resultFromExecute)) {
        fprintf(stderr, "%lu : %lu\n", findMaxFromNFloats(result) + 1,
               findMaxFromNFloats(resultFromExecute) + 1);
        calcSoftmax(resultFromExecute);
        for (size_t j = 0; j < digitNumber; ++j) {
          fprintf(stderr,"%lu.%lu) %f : %f\n", i, j+1, result[j],
                  resultFromExecute[j]);
        }
        fprintf(stderr,"\n");
      }
    }
  }

  void calcSoftmax(float* arr, const size_t& size = 10) {
    size_t max_elem = findMaxFromNFloats(arr, size);
    std::cout << "max_elem " << max_elem << std::endl;
    float result[size];
    float sum_exp = 0.;
    for (size_t i = 0; i < size; ++i) {
      sum_exp += std::exp(arr[i] - arr[max_elem]);
    }

    for (size_t i = 0; i < size; ++i) {
      result[i] = std::exp(arr[i] - arr[max_elem])/sum_exp;
    }

    for (size_t i = 0; i < size; ++i) {
      arr[i] = result[i];
    }
  }

  size_t findMaxFromNFloats(const float* arr, const size_t& arr_size = 10) {
    float max = FLT_MIN;
    size_t max_num = 0;
    for (size_t i = 0; i < arr_size; ++i) {
      if (max < arr[i]) {
        max = arr[i];
        max_num = i;
      }
    }

    return max_num;
  }

  string current_path;
};

TEST_F(WorkflowLoaderTest, MainTest) {
  Mnist();
}

}  // namespace Veles

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "tests/google/src/gtest_main.cc"
#pragma GCC diagnostic pop
