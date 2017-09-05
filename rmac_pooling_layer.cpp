#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/rmac_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
LayerParameter RMACPoolingLayer<Dtype>::GetPoolingParam(const int rmac_level_idx, 
	const int bottom_h, const int bottom_w, const RMACPoolingParameter rmac_param) {
	// rmac_level_idx: 1, 2, 3...
	LayerParameter pooling_param;
	int pad_h = 0;
	int pad_w = 0;
	int kernel_h = 0;
	int kernel_w = 0;
	int stride_h = 0;
	int stride_w = 0;
	if (bottom_w < bottom_h) {
		kernel_w = 2 * bottom_w / (rmac_level_idx + 1);
		kernel_h = kernel_w;
		if (rmac_level_idx == 1)
			stride_w = bottom_w;
		else
			stride_w = static_cast<int>(ceil(static_cast<float>(
				bottom_w - kernel_w) / (rmac_level_idx - 1)));
		stride_h = static_cast<int>(ceil(static_cast<float>(
				bottom_h - kernel_h) / (rmac_level_idx)));
	}
	else if (bottom_w > bottom_h) {
		// std::cout << "Here!" << std::endl;
		kernel_h = 2 * bottom_h / (rmac_level_idx + 1);
		kernel_w = kernel_h;
		// std::cout << "kernel_h, rmac_level_idx: " << kernel_h << ", " << rmac_level_idx << std::endl;
		if (rmac_level_idx == 1)
			stride_h = bottom_h;
		else
			stride_h = static_cast<int>(ceil(static_cast<float>(
				bottom_h - kernel_h) / (rmac_level_idx - 1)));
		stride_w = static_cast<int>(ceil(static_cast<float>(
				bottom_w - kernel_w) / (rmac_level_idx)));
	}
	else {
		kernel_w = 2 * bottom_w / (rmac_level_idx + 1);
		kernel_h = kernel_w;
		if (rmac_level_idx == 1)
			stride_w = bottom_w;
		else
			stride_w = static_cast<int>(ceil(static_cast<float>(
				bottom_w - kernel_w) / (rmac_level_idx - 1)));
		stride_h = stride_w;
	}
	// std::cout << "Hey: Bottom_w, Bottom_h: " << bottom_w << ", " << bottom_h << std::endl;
	// std::cout << "Hey: pad_w, pad_h: " << pad_w << ", " << pad_h << std::endl;
	pooling_param.mutable_pooling_param()->set_pad_w(pad_w);
	pooling_param.mutable_pooling_param()->set_pad_h(pad_h);
	pooling_param.mutable_pooling_param()->set_kernel_w(kernel_w);
	pooling_param.mutable_pooling_param()->set_kernel_h(kernel_h);
	pooling_param.mutable_pooling_param()->set_stride_w(stride_w);
	pooling_param.mutable_pooling_param()->set_stride_h(stride_h);

	switch (rmac_param.pool()) {
	case SPPParameter_PoolMethod_MAX:
		pooling_param.mutable_pooling_param()->set_pool(
		    PoolingParameter_PoolMethod_MAX);
		break;
	case SPPParameter_PoolMethod_AVE:
		pooling_param.mutable_pooling_param()->set_pool(
		    PoolingParameter_PoolMethod_AVE);
		break;
	case SPPParameter_PoolMethod_STOCHASTIC:
		pooling_param.mutable_pooling_param()->set_pool(
		    PoolingParameter_PoolMethod_STOCHASTIC);
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}

	return pooling_param;
}

// TODO
template <typename Dtype>
LayerParameter RMACPoolingLayer<Dtype>::GetGlobalPoolingParam(const int rmac_level_idx, 
	const int bottom_h, const int bottom_w) {
	LayerParameter global_pooling_param;
	int pad_h = 0;
	int pad_w = 0;
	int kernel_h = 0;
	int kernel_w = 0;
	int stride_h = 0;
	int stride_w = 0;
	// global pooling
	if (bottom_w < bottom_h) {
		kernel_w = rmac_level_idx;
		kernel_h = rmac_level_idx + 1;
		stride_w = kernel_w;
		stride_h = kernel_h;
	}
	else if (bottom_w > bottom_h) {
		kernel_w = rmac_level_idx + 1;
		kernel_h = rmac_level_idx;
		stride_w = kernel_w;
		stride_h = kernel_h;
	}
	else {
		kernel_w = rmac_level_idx * rmac_level_idx;
		kernel_h = kernel_w;
		stride_w = kernel_w;
		stride_h = kernel_h;
	}

	global_pooling_param.mutable_pooling_param()->set_pad_w(pad_w);
	global_pooling_param.mutable_pooling_param()->set_pad_h(pad_h);
	global_pooling_param.mutable_pooling_param()->set_kernel_w(kernel_w);
	global_pooling_param.mutable_pooling_param()->set_kernel_h(kernel_h);
	global_pooling_param.mutable_pooling_param()->set_stride_w(stride_w);
	global_pooling_param.mutable_pooling_param()->set_stride_h(stride_h);

	global_pooling_param.mutable_pooling_param()->set_pool(
	    PoolingParameter_PoolMethod_AVE);
	return global_pooling_param;
}

template <typename Dtype>
void RMACPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// std::cout << "RMAC LayerSetup" << std::endl;
	RMACPoolingParameter rmac_param = this->layer_param_.rmac_param();

	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	bottom_h_ = bottom[0]->height();
	bottom_w_ = bottom[0]->width();
	reshaped_first_time_ = false;
	CHECK_GT(bottom_h_, 0) << "Input dim cannot be zero.";
	CHECK_GT(bottom_w_, 0) << "Input dim cannot be zero.";

	rmac_level_ = rmac_param.rmac_level();
	split_top_vec_.clear();
	pooling_bottom_vecs_.clear();
	pooling_layers_.clear();
	pooling_top_vecs_.clear();
	pooling_outputs_.clear();
	global_avg_pooling_layers_.clear();
	global_avg_pooling_top_vecs_.clear();
	global_avg_pooling_outputs_.clear();
	norm_layers_.clear();
	norm_top_vecs_.clear();
	norm_outputs_.clear();
	eltwise_bottom_vec_.clear();
	eltwise_top_vec_.clear();

	// split layer output holders setup
	for (int i = 0; i < rmac_level_; ++i) {
		split_top_vec_.push_back(new Blob<Dtype>());
	}
	// split layer setup
	LayerParameter split_param;
	split_layer_.reset(new SplitLayer<Dtype>(split_param));
	split_layer_->SetUp(bottom, split_top_vec_);

	for (int i = 0; i < rmac_level_; i++) {
		pooling_bottom_vecs_.push_back(new vector<Blob<Dtype>*>);
		pooling_bottom_vecs_[i]->push_back(split_top_vec_[i]);

		pooling_outputs_.push_back(new Blob<Dtype>());
		pooling_top_vecs_.push_back(new vector<Blob<Dtype>*>);
		pooling_top_vecs_[i]->push_back(pooling_outputs_[i]);

		// pooling layer setup
		LayerParameter pooling_param = GetPoolingParam(
			i+1, bottom_h_, bottom_w_, rmac_param);
		pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (
			new PoolingLayer<Dtype>(pooling_param)));
		pooling_layers_[i]->SetUp(*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);

		// global avg pooling layer setup
		global_avg_pooling_outputs_.push_back(new Blob<Dtype>());
		global_avg_pooling_top_vecs_.push_back(new vector<Blob<Dtype>*>);
		global_avg_pooling_top_vecs_[i]->push_back(global_avg_pooling_outputs_[i]);

		LayerParameter global_pooling_param = GetGlobalPoolingParam(
			i+1, bottom_h_, bottom_w_);
		global_avg_pooling_layers_.push_back(shared_ptr<PoolingLayer<Dtype> > (
			new PoolingLayer<Dtype>(global_pooling_param)));
		global_avg_pooling_layers_[i]->SetUp(*pooling_top_vecs_[i], *global_avg_pooling_top_vecs_[i]);

		// normalize layer setup
		norm_outputs_.push_back(new Blob<Dtype>());
		norm_top_vecs_.push_back(new vector<Blob<Dtype>*>);
		norm_top_vecs_[i]->push_back(norm_outputs_[i]);

		LayerParameter norm_param;
		norm_layers_.push_back(shared_ptr<NormalizeLayer<Dtype> > (
			new NormalizeLayer<Dtype>(norm_param)));
		norm_layers_[i]->SetUp(*global_avg_pooling_top_vecs_[i], *norm_top_vecs_[i]);

		// eltwise layer input holders setup
		eltwise_bottom_vec_.push_back(norm_outputs_[i]);
	}
	// eltwise layer setup
	eltwise_top_vec_.push_back(new Blob<Dtype>());

	LayerParameter eltwise_param;
	eltwise_layer_.reset(new EltwiseLayer<Dtype>(eltwise_param));
	eltwise_layer_->SetUp(eltwise_bottom_vec_, eltwise_top_vec_);

	// flatten layer setup
	LayerParameter flatten_param;
	flatten_layer_.reset(new FlattenLayer<Dtype>(flatten_param));
	flatten_layer_->SetUp(eltwise_top_vec_, top);
}

template <typename Dtype>
void RMACPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// std::cout << "Reshape first" << std::endl;
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	// do nothing if bottom shape is unchanged since last reshape
	if (num_ == bottom[0]->num() && channels_ == bottom[0]->channels()
		&& bottom_h_ == bottom[0]->height() && bottom_w_ == bottom[0]->width()
		&& reshaped_first_time_) 
		return;
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	bottom_h_ = bottom[0]->height();
	bottom_w_ = bottom[0]->width();
	reshaped_first_time_ = true;
	RMACPoolingParameter rmac_param = this->layer_param_.rmac_param();
    /////////////// test

	split_layer_->Reshape(bottom, split_top_vec_);
	
	for (int i = 0; i < rmac_level_; ++i) {
		// std::cout << "rmac_level_idx: " << i  << "***********"<< std::endl;
		LayerParameter pooling_param = GetPoolingParam(
			i+1, bottom_h_, bottom_w_, rmac_param);
		pooling_layers_[i].reset(
			new PoolingLayer<Dtype>(pooling_param));
		pooling_layers_[i]->SetUp(
			*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
		pooling_layers_[i]->Reshape(
			*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
		// TODO
		LayerParameter global_pooling_param = GetGlobalPoolingParam(
			i+1, bottom_h_, bottom_w_);
		global_avg_pooling_layers_[i].reset(
			new PoolingLayer<Dtype>(global_pooling_param));
		global_avg_pooling_layers_[i]->SetUp(
			*pooling_top_vecs_[i], *global_avg_pooling_top_vecs_[i]);
		global_avg_pooling_layers_[i]->Reshape(
			*pooling_top_vecs_[i], *global_avg_pooling_top_vecs_[i]);

		norm_layers_[i]->Reshape(*global_avg_pooling_top_vecs_[i], *norm_top_vecs_[i]);		
	}
	eltwise_layer_->Reshape(eltwise_bottom_vec_, eltwise_top_vec_);
	flatten_layer_->Reshape(eltwise_top_vec_, top);
}

template <typename Dtype>
void RMACPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// std::cout << "=======================" << std::endl;
	// std::cout << "Forward_cpu: " << bottom_h_ << ", " << bottom_w_ << std::endl;
	// std::cout << "=======================" << std::endl;
	split_layer_->Forward(bottom, split_top_vec_);
	for (int i = 0; i < rmac_level_; ++i) {
		pooling_layers_[i]->Forward(
			*pooling_bottom_vecs_[i], *pooling_top_vecs_[i]);
		global_avg_pooling_layers_[i]->Forward(
			*pooling_top_vecs_[i], *global_avg_pooling_top_vecs_[i]);
		norm_layers_[i]->Forward(
			*global_avg_pooling_top_vecs_[i], *norm_top_vecs_[i]);		
	}
	eltwise_layer_->Forward(eltwise_bottom_vec_, eltwise_top_vec_);
	flatten_layer_->Forward(eltwise_top_vec_, top);
}

template <typename Dtype>
void RMACPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0])
		return;

	flatten_layer_->Backward(
		top, propagate_down, eltwise_top_vec_);
	vector<bool> eltwise_propagate_down(rmac_level_, true);
	eltwise_layer_->Backward(
		eltwise_top_vec_, eltwise_propagate_down, eltwise_bottom_vec_);
	for (int i = 0; i < rmac_level_; ++i) {
		norm_layers_[i]->Backward(
			*norm_top_vecs_[i], propagate_down, *global_avg_pooling_top_vecs_[i]);
		global_avg_pooling_layers_[i]->Backward(
			*global_avg_pooling_top_vecs_[i], propagate_down, *pooling_top_vecs_[i]);
		pooling_layers_[i]->Backward(
			*pooling_top_vecs_[i], propagate_down, *pooling_bottom_vecs_[i]);
	}
	split_layer_->Backward(split_top_vec_, propagate_down, bottom);
}

INSTANTIATE_CLASS(RMACPoolingLayer);
REGISTER_LAYER_CLASS(RMACPooling);

} //namespace caffe
