#ifndef CBIR_H
#define CBIR_H

#include <string>
#include <vector>
using namespace std;

class Cbir {
public:
	Cbir();
	~Cbir();	

	////<<<< 库特征提取接口
	/// @brief 库特征提取,保存降维特征和PCA模型
	/// @params image_list_file 库图片列表路径
	/// @params result_list_file 要保存的库列表路径 
	/// @params result_feats_file 要保存的库特征路径
	/// @params result_pca_model_file 要保存的pca模型路径
	void cbir_predict_and_save_db(const char* image_list_file,
								  const char* result_list_file,
							      const char* result_feats_file,
							      const char* result_pca_model_file);


	////<<<< 查询接口
	/// @brief 提取查询图片特征,搜索得到结果
	/// @params search_list_file 要查询的图片列表文件
	/// @params db_list_file 库图片列表路径
	/// @params db_feat_file 库特征文件路径
	/// @params pca_model_file pca模型路径
	/// @params search_result_folder 保存查询图片的搜索结果文件夹
	void cbir_search_and_save_result(const char* search_list_file,
									 const char* db_list_file,
									 const char* db_feat_file,
									 const char* pca_model_file,
									 const char* search_result_folder);

private:
	void cbir_init();
	void predict(const char* image_list_file, 
			 vector<string> &result_list_vec,
			 vector<vector<float> > &result_feats_vec);
	// dataset
	void save_db(const vector<string> &result_list_vec,
			 	 const vector<vector<float> > &result_feats_vec,
			 	 const char* result_list_file,
				 const char* result_feats_file,
				 const char* result_pca_model_file);
	void save_result(const vector<string> &search_list_vec,
				     const vector<vector<float> > &search_feats_vec,
				     const char* db_list_file,
				     const char* db_feat_file,
				     const char* pca_model_file,
				     const char* search_result_folder);	
	
};

#endif
