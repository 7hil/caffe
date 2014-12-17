#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void OnehotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const OnehotParameter& param = this->layer_param_.onehot_param();
  this->invert_coding_ = param.invert_coding();
  this->dim_ = param.dim();
  CHECK(this->dim_ > 0)
      << "output dimension should larger than 0.";
  CHECK(bottom[0]->channels()==1 && bottom[0]->height()==1 &&
        bottom[0]->width()==1)
      << "shape of bottom[0] should be (num,1,1,1)";
}

template <typename Dtype>
void OnehotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  top[0]->Reshape(bottom[0]->num(), this->dim_, 1, 1);
}

template <typename Dtype>
void OnehotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype pos = this->invert_coding_?Dtype(0):Dtype(1);
  Dtype neg = this->invert_coding_?Dtype(1):Dtype(0);
  caffe_set(top[0]->count(), neg, top_data);
  for (int n = 0; n < bottom[0]->num(); n++) {
    int index = static_cast<int>(bottom[0]->data_at(n, 0, 0, 0));
    CHECK(index >= 0 && index < this->dim_)
        << "wrong index " << index << " from sample " << n;
    caffe_set(1, pos, top_data+top[0]->offset(n)+index);
  }
}

#ifdef CPU_ONLY
STUB_GPU(OnehotLayer);
#endif

INSTANTIATE_CLASS(OnehotLayer);
REGISTER_LAYER_CLASS(ONEHOT, OnehotLayer);
}  // namespace caffe
