#include <string>
#include <caffe/caffe.hpp>
#include <caffe/util/upgrade_proto.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "common.h"

using namespace caffe;
using namespace std;

class Process {
	
public:
	Process(const string& model_def, const string& weight_file, 
			const vector<float> mean_vec, 
			const vector<int> multiscale_vec,
			const vector<string> layers_vec);
	~Process();

	// predict a single image.
	vector<pair<string, vector<float> > > predict(const string& image_path,
			const int cpu_idx);
	vector<vector<float> > pca(const vector<vector<float> > x,
			                   vector<vector<float> > &w);
	Eigen::MatrixXf pca(const vector<vector<float> > x,
						const char* pca_mode_file);
	vector<pair<float, string> > calcu_cos_dist_and_sort(Eigen::VectorXf search_v,
														vector<string> db_list_vec,
							 							vector<vector<float> > db_feat_vec);


private:
	void init(const string& model_def, const string& weight_file, 
			  const vector<float> mean_vec,
			  const vector<int> multiscale_vec,
			  const vector<string> layers_vec);

	bool mean_and_resize(const string& image_path,
					 const int scale,
					 const int cpu_idx,
					 vector<Blob<float>*>& bottom);

	Eigen::MatrixXf normalize(Eigen::MatrixXf x);

	// std::vector<float> rmac(boost::shared_ptr<Blob<float> >& blob);

	// caffe net, for forwarding the computation.
	vector<Net<float>*> caffe_net_vec_;
	vector<vector<int> > shape_;
	// image preprocess, for preprocess the image.
	vector<float> mean_vec_;
	vector<int> multiscale_vec_;
	vector<string> layers_vec_;
	// rmac setting
	int L;

};
