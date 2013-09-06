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

#include <unistd.h>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <string>
#include <veles/workflow_loader.h>
#include <gtest/gtest.h>
#include "src/all2all_linear.h"
#include "src/all2all_tanh.h"

using std::string;
using std::static_pointer_cast;

namespace Veles {

class WorkflowLoaderTest: public ::testing::Test {
 public:
  WorkflowLoaderTest() {
    current_path = "";
    char currentPath[FILENAME_MAX];

    if (!getcwd(currentPath, sizeof(currentPath))) {
      fprintf(stderr, "Can't locate current directory\n");
    } else {
      current_path = currentPath;
    }
    // Check existence of the archive directory
    auto try_path = current_path + "/../../tests/workflow_files/";
    if (access(try_path.c_str(), 0) != -1) {
      current_path = try_path;
      return;
    }
    try_path = current_path + "/../tests/workflow_files/";
    if (access(try_path.c_str(), 0) != -1) {
      current_path = try_path;
      return;
    }
    try_path = current_path + "/tests/workflow_files/";
    if (access(try_path.c_str(), 0) != -1) {
      current_path = try_path;
      return;
    }
    fprintf(stderr, "%s", current_path.c_str());
    current_path = "";
  }

  void ChronoTest() {
    if (current_path == string("")) {
      FAIL() << "Can't find folder workflow_desc_files";
    }
    WorkflowLoader testWorkflowLoader;

    string temp = current_path + "workflow2013-09-03T23:19:38.323304.tar.gz";

    ASSERT_NO_THROW(testWorkflowLoader.Load(temp));

    Workflow testWorkflow;
    ASSERT_NO_THROW(testWorkflow = testWorkflowLoader.GetWorkflow());


    const size_t imgNumber = 60;
    const size_t pxNumber = 784;
    const size_t outputNumber = 100;

    typedef std::chrono::high_resolution_clock Time;

    auto t1 = Time::now();

    time_t tt1 = time(0);
    float resultFromExecute[outputNumber];
    for (size_t i = 0; i < imgNumber; ++i) {
      auto begin = static_cast<float*>(
          testWorkflowLoader.workflow_desc_.Units.at(0).Properties.at("input").get());
      float array[pxNumber];
      for (size_t j = 0; j < pxNumber; ++j) {
        array[j] = begin[j + pxNumber*i];
      }
      testWorkflow.Execute(array, array + pxNumber - 1, resultFromExecute);
    }
    time_t tt2 = time(0);
    fprintf(stderr, "time_t diff = %ld", tt2 - tt1);
    auto t2 = Time::now();

    std::chrono::duration<double> time_span = std::chrono::duration_cast<
        std::chrono::duration<double>>(t2 - t1);
    fprintf(stderr, "count %lf\n", time_span.count());
    double result = std::chrono::duration_cast<std::chrono::duration<double,
        std::ratio<1>>>
        (t2-t1).count();

    fprintf(stderr, "Total duration: %lf\n", result);
    fprintf(stderr, "One image: %lf\n", result/imgNumber);
  }

  void Mnist() {
    if (current_path == string("")) {
      FAIL() << "Can't find folder workflow_desc_files";
    }
    WorkflowLoader testWorkflowLoader;

    string temp = current_path + "workflow.tar.gz";
    ASSERT_NO_THROW(testWorkflowLoader.Load(temp));

    Workflow testWorkflow;
    ASSERT_NO_THROW(testWorkflow = testWorkflowLoader.GetWorkflow());

    // Get 60 images from WorkflowLoader: unit "All2AllTanh", parameter "input"
    // Image = 28*28 pixels, each pixel is represented by float
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
      testWorkflow.Execute(array, array + pxNumber - 1, resultFromExecute);

      if (findMaxFromNFloats(result) != findMaxFromNFloats(resultFromExecute)) {
        fprintf(stderr, "%zu : %zu\n", findMaxFromNFloats(result) + 1,
               findMaxFromNFloats(resultFromExecute) + 1);
        calcSoftmax(resultFromExecute);
        for (size_t j = 0; j < digitNumber; ++j) {
          fprintf(stderr,"%zu.%zu) %f : %f\n", i, j+1, result[j],
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

/*
TEST_F(WorkflowLoaderTest, Chrono) {
  ChronoTest();
}
*/

}  // namespace Veles

#include "tests/google/src/gtest_main.cc"
