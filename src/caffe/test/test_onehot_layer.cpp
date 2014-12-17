#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class OnehotLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OnehotLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    dim = 6;
    // Initialize the index
    Dtype* index_data = blob_bottom_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_->count(); ++i) {
      index_data[i] = i % dim;
    }
    blob_bottom_vec.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    OnehotParameter* onehot_param = layer_param.mutable_onehot_param();
    onehot_param->set_dim(dim);
  }

  virtual ~OnehotLayerTest() {
    delete blob_bottom_; delete blob_top_;
  }
  int dim;
  LayerParameter layer_param;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OnehotLayerTest, TestDtypesAndDevices);

TYPED_TEST(OnehotLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  OnehotLayer<Dtype> layer(this->layer_param);
  layer.SetUp(this->blob_bottom_vec, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->dim);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(OnehotLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  OnehotLayer<Dtype> layer(this->layer_param);
  layer.SetUp(this->blob_bottom_vec, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->num(); ++n) {
    int index = this->blob_bottom_->data_at(n, 0, 0, 0);
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      // TODO: test inverted coding
      EXPECT_EQ(this->blob_top_->data_at(n, c, 0, 0),
          Dtype(index==c));
    }
  }
}

}  // namespace caffe
