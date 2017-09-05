#include "process.h"
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <iostream>
#include <fstream>

using namespace std;

Process::Process(const string& model_def, const string& weight_file, 
				 const vector<float> mean_vec, 
				 const vector<int> multiscale_vec,
				 const vector<string> layers_vec) {
	init(model_def, weight_file, mean_vec, multiscale_vec, layers_vec);	
}

Process::~Process() {
	for (int i = 0; i < ITER_TASK; ++i) {
		if (this->caffe_net_vec_[i]) {
			delete this->caffe_net_vec_[i];
			this->caffe_net_vec_[i] = NULL;
		}
	}
}

void Process::init(const string& model_def, const string& weight_file, 
					const vector<float> mean_vec,
					const vector<int> multiscale_vec,
					const vector<string> layers_vec) {
	// init
	this->multiscale_vec_ = multiscale_vec;
	this->layers_vec_ = layers_vec;
	this->mean_vec_ = mean_vec;
	this->L = 3;

	// read param from prototxt
	NetParameter param;
	ReadNetParamsFromTextFileOrDie(model_def, &param);
	// caffe net init
	// Caffe::set_mode(Caffe::CPU);
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(0);
	for (int i = 0; i < ITER_TASK; ++i) {
		vector<int> s;
		this->shape_.push_back(s);
		this->caffe_net_vec_.push_back(new Net<float>(model_def, caffe::TEST));
		this->caffe_net_vec_[i]->CopyTrainedLayersFrom(weight_file);
	}

}

bool Process::mean_and_resize(const string& image_path,
						 		const int scale,
						 		const int cpu_idx,
						 		vector<Blob<float>*>& bottom) {

	cv::Mat img = cv::imread(image_path, 1);
	if (img.empty()) {
		cout << "image empty! " << image_path << endl;
		return 0;
	}
	const int img_channels = img.channels();
	const int img_height = img.rows;
	const int img_width = img.cols;

    float ratio = 0.;
	if (img_height < img_width) {
		ratio = scale * 1.0 / img_width;
		if (img_height * ratio < 224)
			ratio = 224.0 / img_height;
	}
	else {
		ratio = scale * 1.0 / img_height;
		if (img_width * ratio < 224)
			ratio = 224.0 / img_width;
	}

	const int new_height = img_height * ratio;
	const int new_width = img_width * ratio;
	// const int new_height = 224;
	// const int new_width = 224;
	cout << new_height << " " << new_width << endl;
	// blob reshape
	this->shape_[cpu_idx].push_back(1); // n
	this->shape_[cpu_idx].push_back(img_channels); // c
	this->shape_[cpu_idx].push_back(new_height); // h
	this->shape_[cpu_idx].push_back(new_width); // w
	bottom[0]->Reshape(this->shape_[cpu_idx]);
	float* bottom_data = bottom[0]->mutable_cpu_data();

	cv::resize(img, img, cv::Size(new_height, new_width));
	int index = 0;
	for (int h = 0; h < new_height; ++h) {
		const uchar* ptr = img.ptr<uchar>(h);
		int img_idx = 0;
		for (int w = 0; w < new_width; ++w) {
			for (int c = 0; c < img_channels; ++c) {
				index = (c * new_height + h) * new_width + w;
				float pixel = static_cast<float>(ptr[img_idx++]);
				bottom_data[index] = pixel - mean_vec_[c];
			}
		}
	}
	// clear element but remain memory
	this->shape_[cpu_idx].clear();
	return 1;
}

vector<pair<string, vector<float> > > Process::predict(const string& image_path,
	const int cpu_idx) {
	struct timeval s_time, e_time;
	gettimeofday(&s_time, NULL);
	vector<Blob<float>*> input_layer = this->caffe_net_vec_[cpu_idx]->input_blobs();

	vector<pair<string, vector<float> > > single_result;
	vector<float> feats_vec(FEATURE_LENGTH, 0.);
	for (int i = 0; i < multiscale_vec_.size(); ++i) {
		// fullfill input data layer and pre-process
		if (!mean_and_resize(image_path, multiscale_vec_[i], cpu_idx, input_layer))
			exit(1);

		// forward multi_layer
		this->caffe_net_vec_[cpu_idx]->Forward();
		vector<boost::shared_ptr<Blob<float> > > blobs = this->caffe_net_vec_[cpu_idx]->blobs();
		vector<string> blob_names = this->caffe_net_vec_[cpu_idx]->blob_names();
		// blob_names.size(): 320;
		// blob_names[319]: multi_layer_concat_norm
		// cout << "the last layer: " << blob_names[LAYER_IDX] << endl;
		const float* feature_data = blobs[LAYER_IDX]->cpu_data();
		
		for (int j = 0; j < FEATURE_LENGTH; ++j) {
			feats_vec[j] += feature_data[j];
		}
	}
	single_result.push_back(make_pair<string, vector<float> >(image_path, feats_vec));

	gettimeofday(&e_time, NULL);
	double time_consumed;
	time_consumed = (double)(1000000 * (e_time.tv_sec - s_time.tv_sec) +
					(e_time.tv_usec - s_time.tv_usec));
	time_consumed /= 1000000;
	cout << "\npredict time:" << time_consumed << endl;
	return single_result;
}

vector<vector<float> > Process::pca(const vector<vector<float> > x,
	vector<vector<float> > &w) {
	cout << "\nThis is P! C! A!" << endl;
	int rows = x.size(); // samples
	int cols = x[0].size(); // dims
	// assert samples > dims
	bool flag = false;
	if (rows <= cols) {
		rows = cols + 1;
		flag = true; // Oops..here made a flag
	}
	
	Eigen::MatrixXf ori_x(x.size(), x[0].size());
	Eigen::MatrixXf X(rows, cols);
	cout << "rows: " << rows << ", cols: " << cols << endl;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			X(i, j) = x[i%x.size()][j];
		}
	}
	if (flag) {
		for (int i = 0; i < x.size(); ++i) {
			for (int j = 0; j < x[0].size(); ++j) {
				ori_x(i, j) = x[i][j];
			}
		}	
	}
	struct timeval s_time, e_time;
	gettimeofday(&s_time, NULL);
	// // start
	// // mean center
	// Eigen::MatrixXf centered = X.rowwise() - X.colwise().mean();
	// // normalize
	// X = normalize(centered);
	// // compute covariance matrix
	// Eigen::MatrixXf cov = X.adjoint() * X;
	// cov = cov / (X.rows() - 1);
	// Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
	// // Normalize eigenvalues to make them represent percentages.
	// Eigen::VectorXf normalizedEigenValues =  eig.eigenvalues() / eig.eigenvalues().sum();
	// // Get the major eigenvectors and omit the others.
	// Eigen::MatrixXf evecs = eig.eigenvectors();
	// Eigen::MatrixXf W = evecs.rightCols(FEATURE_PCA_LENGTH);

	// // Map the dataset in the new two dimensional space.
	// Eigen::MatrixXf projected = X * W;
	// if (flag) {
	// 	Eigen::MatrixXf ori_x_aligned = ori_x.rowwise() - ori_x.colwise().mean();
	// 	ori_x = normalize(ori_x_aligned);
	// 	projected = ori_x * W;
	// }
	// projected = normalize(projected);
	// // end

	X = normalize(X);
	Eigen::MatrixXf X_aligned = X.rowwise() - X.colwise().mean();
	// X_aligned = normalize(X_aligned);
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(X_aligned, Eigen::ComputeThinV);
	Eigen::MatrixXf W = svd.matrixV().leftCols(FEATURE_PCA_LENGTH);
	Eigen::MatrixXf projected = X_aligned * W;
	if (flag) {
		ori_x = normalize(ori_x);
		Eigen::MatrixXf ori_x_aligned = ori_x.rowwise() - ori_x.colwise().mean();
		// ori_x_aligned = normalize(ori_x_aligned);
		projected = ori_x_aligned * W;
	}
	projected = normalize(projected);
	gettimeofday(&e_time, NULL);
	double time_consumed;
	time_consumed = (double)(1000000 * (e_time.tv_sec - s_time.tv_sec) +
					(e_time.tv_usec - s_time.tv_usec));
	time_consumed /= 1000000;
	cout << "\nP! C! A! time: " << time_consumed << " s" << endl << endl;
	cout << "projected.rows: " << projected.rows() << ", projected.cols: " << projected.cols() << endl;
	cout << "W.rows: " << W.rows() << ", W.cols: " << W.cols() << endl;

 	for (int i = 0; i < W.rows(); ++i) {
		for (int j = 0; j < W.cols(); ++j) {
			w[i][j] = W(i, j);
		}
	}
	vector<vector<float> > result;
 	for (int i = 0; i < projected.rows(); ++i) {
 		vector<float> r;
 		result.push_back(r);
		for (int j = 0; j < projected.cols(); ++j) {
			result[i].push_back(projected(i, j));
		}
	}
	return result;
}

Eigen::MatrixXf Process::pca(const vector<vector<float> > x,
									const char* pca_mode_file) {

	ifstream fin(pca_mode_file, ifstream::binary);
	if (!fin) {
		cout << "[ERROR] pca model file not exists!\n";
		exit(1);
	}
	Eigen::MatrixXf W(FEATURE_LENGTH, FEATURE_PCA_LENGTH);
	// cout << W.rows() << " " << W.cols();
	for (int i = 0; i < FEATURE_LENGTH; ++i) {
		for (int j = 0; j < FEATURE_PCA_LENGTH; ++j) {
			fin.read((char*)&W(i, j), sizeof(float)*1);
		}
	}
	fin.close();
	cout << "\nWow" << endl;
	Eigen::MatrixXf X(x.size(), x[0].size());
	for (int i = 0; i < x.size(); ++i) {
		for (int j = 0; j < x[0].size(); ++j) {
			X(i, j) = x[i][j];
		}
	}
	// // start
	// Eigen::MatrixXf X_aligned = X.rowwise() - X.colwise().mean();
	// X = normalize(X_aligned);
	// Eigen::MatrixXf projected = X * W;
	// projected = normalize(projected);
	// // end
	X = normalize(X);
	Eigen::MatrixXf aligned = X.rowwise() - X.colwise().mean();
	// aligned = normalize(aligned);
	Eigen::MatrixXf projected = aligned * W;
	projected = normalize(projected);


	return projected;
}

vector<pair<float, string> > Process::calcu_cos_dist_and_sort(Eigen::VectorXf search_v,
							 vector<string> db_list_vec,
							 vector<vector<float> > db_feat_vec) {
	vector<pair<float, string> > result;
	for (int i = 0; i < db_feat_vec.size(); ++i) {
		Eigen::VectorXf db_feat_v(db_feat_vec[i].size());
		for (int j = 0; j < db_feat_vec[i].size(); ++j) {
			db_feat_v(j) = db_feat_vec[i][j];
		}
		// calcu dist
		float dist = search_v.dot(db_feat_v);
		result.push_back(make_pair<float, string>(dist, db_list_vec[i]));
	}
	// sort, in descending order
	sort(result.begin(), result.end());
	return result;
}

Eigen::MatrixXf Process::normalize(Eigen::MatrixXf x) {

	Eigen::MatrixXf res(x.rows(), x.cols());
	Eigen::VectorXf v(x.cols());
	for (int i = 0; i < x.rows(); ++i) {
		v = x.row(i);
		res.row(i) = v.normalized();
	}

	return res;
}

