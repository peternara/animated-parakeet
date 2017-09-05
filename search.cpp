#include "cbir.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char** argv) {
	if (argc != 6) {
		cout << "[ERROR] Usage: image_list db_folder\n\n";
		exit(1);
	}

	char* search_list_file = argv[1];
	char* db_list_file = argv[2];
	char* db_feat_file = argv[3];
	char* db_pca_model_file = argv[4];
	char* search_result_folder = argv[5];

	// init
	Cbir* cbir = new Cbir();
	// db includes: image list, image feature(pcaed), pca model
	cbir->cbir_search_and_save_result(search_list_file,
									  db_list_file,
									  db_feat_file,
									  db_pca_model_file,
									  search_result_folder);
	//destory
	cbir->~Cbir();
	delete cbir;
	cbir = NULL;

	return 0;
}
