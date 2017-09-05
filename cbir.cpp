#include "cbir.h"
#include "common.h"
#include "process.h"
#include <string>
#include <vector>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>

using namespace std;
Process* cbir_process;

Cbir::Cbir() {
	cbir_init();
}

Cbir::~Cbir() {
	if (cbir_process) {
		cbir_process->~Process();
		cbir_process = NULL;
	}
}

void Cbir::cbir_init() {
	struct timeval s_time, e_time;
	gettimeofday(&s_time, NULL);
	vector<float> mean_vec;
	mean_vec.push_back(MEAN_B);
	mean_vec.push_back(MEAN_G);
	mean_vec.push_back(MEAN_R);

	vector<int> multiscale_vec;
	multiscale_vec.push_back(MULTISCALE_S);
	multiscale_vec.push_back(MULTISCALE_M);
	multiscale_vec.push_back(MULTISCALE_L);

	vector<string> layers_vec;
	layers_vec.push_back(LAYER_NAME);
	cbir_process = new Process(MODEL_DEF, WEIGHT_FILE, mean_vec, multiscale_vec, layers_vec);

	gettimeofday(&e_time, NULL);
	double time_consumed;
	time_consumed = (double)(1000000 * (e_time.tv_sec - s_time.tv_sec) + 
						(e_time.tv_usec - s_time.tv_usec));
	time_consumed /= 1000000;
	cout << "Init time: " << time_consumed << endl;

}

////////////////////////////////////////////////////////////
/// gen dataset: feature.txt, feature.bin, pca_model.bin
////////////////////////////////////////////////////////////
void Cbir::cbir_predict_and_save_db(const char* image_list_file,
							  const char* result_list_file,
							  const char* result_feats_file,
							  const char* result_pca_model_file) {
	vector<string> result_list_vec;
	vector<vector<float> > result_feats_vec;
	// predict image list
	predict(image_list_file,
			result_list_vec,
			result_feats_vec);
	// pca and save db
	save_db(result_list_vec,
			result_feats_vec,
			result_list_file,
			result_feats_file,
			result_pca_model_file);
}

void Cbir::predict(const char* image_list_file, 
			 vector<string> &result_list_vec,
			 vector<vector<float> > &result_feats_vec) {
	const int iter_task = ITER_TASK;
	string image_name_arr[iter_task]; 
	vector<string> tmp_list_vec(iter_task);
	vector<vector<float> > tmp_feats_vec(iter_task, vector<float>(FEATURE_LENGTH));
	ifstream fin(image_list_file);
	if (!fin) {
		cout << "[ERROR] false image list path\n";
		cout << image_list_file;
		return;
	}

	int count = 0;
	while(1) {
		int task_num = 0;
		for (int task_idx = 0; task_idx < iter_task; ++ task_idx) {
			if (!(fin >> image_name_arr[task_idx]))
				break;
			++ task_num;
		}
// #pragma omp parallel for num_threads(ITER_TASK) schedule(dynamic)
		for (int task_idx = 0; task_idx < task_num; ++ task_idx) {
			cout << "************************************" << endl;
			cout << count << " " << image_name_arr[task_idx] << endl;
			vector<pair<string, vector<float> > > single_result = cbir_process->predict(image_name_arr[task_idx], task_idx);
			tmp_list_vec[task_idx] = single_result[0].first;
			tmp_feats_vec[task_idx] = single_result[0].second;
			count += 1;
		}

		// add result
		for (int task_idx = 0; task_idx < task_num; ++ task_idx) {
			result_list_vec.push_back(tmp_list_vec[task_idx]);
			result_feats_vec.push_back(tmp_feats_vec[task_idx]);
		}
		if (task_num != iter_task)
			break;
	}
}

void Cbir::save_db(const vector<string> &result_list_vec,
			       const vector<vector<float> > &result_feats_vec,
				   const char* result_list_file,
				   const char* result_feats_file,
				   const char* result_pca_model_file) {

	ofstream fout_list(result_list_file);
	ofstream fout_feature(result_feats_file, ios::out|ios::binary);
	ofstream fout_pca_model(result_pca_model_file, ios::out|ios::binary);

	// gen result
	// pca_model_vec:(3840, 512)
	vector<vector<float> > result_pca_model_vec(FEATURE_LENGTH, vector<float>(FEATURE_PCA_LENGTH));
	// PCA!
	vector<vector<float> > result_pcaed_feats_vec = cbir_process->pca(result_feats_vec, result_pca_model_vec);
	// (num, 3840)
	cout << "result_feats_vec.size(): " << result_feats_vec.size() << " " << result_feats_vec[0].size() << endl;
	// (num, 512)
	cout << "result_pcaed_feats_vec.size(): " << result_pcaed_feats_vec.size() << " " << result_pcaed_feats_vec[0].size() << endl;
	
	// save dataset: list, pcaed features, pca model
	// list
	for (int i = 0; i < result_list_vec.size(); ++i) {
		fout_list << result_list_vec[i] << "\n";
	}
	fout_list.close();
	// pcaed features
	for (int i = 0; i < result_pcaed_feats_vec.size(); ++i) {
		for (int j = 0; j < result_pcaed_feats_vec[0].size(); ++j) {
			fout_feature.write((char*)&result_pcaed_feats_vec[i][j], 1*sizeof(float));
		}
	}
	fout_feature.close();
	// pca model
	for (int i = 0; i < FEATURE_LENGTH; ++i) {
		for (int j = 0; j < FEATURE_PCA_LENGTH; ++j) {
			fout_pca_model.write((char*)&result_pca_model_vec[i][j], 1*sizeof(float));
		}
	}
	fout_pca_model.close();

	cout << "database saved successfully! " << endl;
}


////////////////////////////////////////////////////////////
/// image retrieval
////////////////////////////////////////////////////////////
void Cbir::cbir_search_and_save_result(const char* search_list_file,
									   const char* db_list_file,
									   const char* db_feat_file,
									   const char* pca_model_file,
									   const char* search_result_folder) {
	vector<string> search_list_vec;
	vector<vector<float> > search_feats_vec;
	// predict image list
	predict(search_list_file,
			search_list_vec,
			search_feats_vec);
	// process result
	save_result(search_list_vec,
				search_feats_vec,
				db_list_file,
				db_feat_file,
				pca_model_file,
				search_result_folder);
	cout << "\nSearch finish!" << endl;;
}

void Cbir::save_result(const vector<string> &search_list_vec,
					   const vector<vector<float> > &search_feats_vec,
					   const char* db_list_file,
					   const char* db_feat_file,
					   const char* pca_model_file,
					   const char* search_result_folder) {
	// (1) load db
	vector<string> db_list_vec;
	ifstream fin_list(db_list_file);
	if (!fin_list) {
		cout << "[ERROR]read db_list_file failed...";
		exit(1);
	}
	string str;
	while(getline(fin_list, str))
		db_list_vec.push_back(str);
	fin_list.close();

	vector<vector<float> > db_feat_vec(db_list_vec.size(), vector<float>(FEATURE_PCA_LENGTH)); 
	ifstream fin_feats(db_feat_file, ios::out|ios::binary);
	if (!fin_feats) {
		cout << "[ERROR]read db_feat_file failed...";
		exit(1);
	}
	for (int i = 0; i < db_feat_vec.size(); ++i) {
		for (int j = 0; j < db_feat_vec[0].size(); ++j) {
			fin_feats.read((char*)&db_feat_vec[i][j], 1*sizeof(float));
		}
	}
	fin_feats.close();

	// pca feats, calcu cosine dist, sort, save result
	Eigen::MatrixXf search_pcaed_feats_matrix;
	// (2) pca
	// search_pcaed_feats_matrix: (search_num, 512)
	cout << "\nstart P!C!A!\n";
	search_pcaed_feats_matrix = cbir_process->pca(search_feats_vec, pca_model_file);
	// (3) calcu cosine dist, sort
	for (int i = 0; i < search_pcaed_feats_matrix.rows(); ++i) {
		cout << "processing search image id: " << i << endl;
		vector<pair<float, string> > single_sorted_result;
		single_sorted_result = cbir_process->calcu_cos_dist_and_sort(search_pcaed_feats_matrix.row(i),
													 db_list_vec,
													 db_feat_vec);
		// (4) save result
		string single_search_path = search_list_vec[i];
		vector<string> fields;
		boost::algorithm::split(fields, single_search_path, boost::algorithm::is_any_of("/"));
		string search_image_name = fields[fields.size()-1]; // Bridge_III_9.jpg
		fields.clear();
		boost::algorithm::split(fields, search_image_name, boost::algorithm::is_any_of("."));
		string search_image_name_s = fields[0]; // Bridge_III_9
		fields.clear();

		string search_result_folder_s = search_result_folder;
		string search_result_path = search_result_folder_s + "/" + search_image_name_s + ".txt";
		ofstream fout(search_result_path.c_str(), ios::out);
		for (int j = single_sorted_result.size()-1; j >= 0 ; --j) {
			vector<string> v;
			boost::algorithm::split(v, single_sorted_result[j].second, boost::algorithm::is_any_of("/"));
			string result_name = v[v.size()-1];
			v.clear();
			string line = result_name +
						  " " +
						  boost::lexical_cast<string>(single_sorted_result[j].first) +
						  "\n";
			fout << line;
		}
		fout.close();
	}
}

