#ifndef COMMON_H
#define COMMON_H

#define MODEL_DEF "./models/deploy_resnet152-places365.rmaclayer.prototxt"
#define WEIGHT_FILE "./models/resnet152_places365.caffemodel"
#define FEATURE_LENGTH 3840
#define FEATURE_PCA_LENGTH 512
#define LAYER_NAME "multi_layer_concat_norm"
#define LAYER_IDX 319
#define MEAN_B 104
#define MEAN_G 112.5
#define MEAN_R 116.7
#define MULTISCALE_S 550
#define MULTISCALE_M 800
#define MULTISCALE_L 1050
#define ITER_TASK 1


#endif
