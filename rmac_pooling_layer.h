#ifndef CAFFE_RMAC_POOLING_LAYER_HPP_
#define CAFFE_RMAC_POOLING_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype> 
class RMACPoolingLayer : public Layer<Dtype> {
	public:
		explicit RMACPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);		

		virtual inline const char* type() const { return "RMACPooling"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		// calculates the kernel and stride dimensions for the pooling layer,
		// returns a correctly configured LayerParameter for a PoolingLayer
		virtual LayerParameter GetPoolingParam(const int rmac_level_idx, 
			const int bottom_h, const int bottom_w, const RMACPoolingParameter rmac_param);
		virtual LayerParameter GetGlobalPoolingParam(const int rmac_level_idx, 
			const int bottom_h, const int bottom_w);

		int rmac_level_; // default: 3
		int bottom_h_, bottom_w_;
		int num_;
		int channels_;
		int pad_h_, pad_w_;
		bool reshaped_first_time_;

		/// the internal Split layer that feeds the pooling layers
		shared_ptr<SplitLayer<Dtype> > split_layer_;
		/// top vector holder used in call to the underlying SplitLayer::Forward
		vector<Blob<Dtype>*> split_top_vec_;
		/// bottom vector holder used in call to the underlying PoolingLayer::Forward
		vector<vector<Blob<Dtype>*>*> pooling_bottom_vecs_;
		/// the internal Pooling layers of different kernel sizes
		vector<shared_ptr<PoolingLayer<Dtype> > > pooling_layers_;
		/// top vector holders used in call to the underlying PoolingLayer::Forward
		vector<vector<Blob<Dtype>*>*> pooling_top_vecs_;
		/// pooling_outputs stores the outputs of the PoolingLayers
		vector<Blob<Dtype>*> pooling_outputs_;

		vector<shared_ptr<PoolingLayer<Dtype> > > global_avg_pooling_layers_;
		vector<vector<Blob<Dtype>*>*> global_avg_pooling_top_vecs_;
		vector<Blob<Dtype>*> global_avg_pooling_outputs_;

		vector<shared_ptr<NormalizeLayer<Dtype> > > norm_layers_;
	    vector<vector<Blob<Dtype>*>*> norm_top_vecs_;
	    vector<Blob<Dtype>*> norm_outputs_;

	    ///////////////////////////////  
	    vector<Blob<Dtype>*> eltwise_bottom_vec_;
	    shared_ptr<EltwiseLayer<Dtype> > eltwise_layer_;
	    vector<Blob<Dtype>*> eltwise_top_vec_;

	    shared_ptr<FlattenLayer<Dtype> > flatten_layer_;
};

} // namespace caffe

#endif // CAFFE_RMAC_POOLING_LAYER_HPP_
