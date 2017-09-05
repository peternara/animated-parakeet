#include "cbir.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char** argv) {
	if (argc != 5) {
		cout << "[ERROR] Usage: image_list result_list_file result_feats_file result_pca_model_file\n\n";
		exit(1);
	}

	char* image_list = argv[1];
	char* result_list_file = argv[2];
	char* result_feats_file = argv[3];
	char* result_pca_model_file = argv[4];

	// init
	Cbir* cbir = new Cbir();
	// db includes: image list, image feature(pcaed), pca model
	cbir->cbir_predict_and_save_db(image_list, result_list_file, result_feats_file, result_pca_model_file);
	//destory
	cbir->~Cbir();

	delete cbir;
	cbir = NULL;

	return 0;
}
