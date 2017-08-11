#!/usr/bin/env python
# encoding: utf-8

import cv2
import matplotlib as mpl
mpl.use('Agg')
import caffe
import os, sys, h5py, timeit, glob
import numpy as np
from shutil import copyfile
from PIL import Image

from sklearn.preprocessing import normalize as sknormalize
from extract_features import extract_fc_features, opencv_format_img_for_vgg
from sklearn.decomposition import PCA
from multi_layer_rmac_feats_extract import ImageHelper
from crow import compute_crow_spatial_weight, compute_crow_channel_weight

def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)

def run_feature_processing_pipeline(features, d=128, whiten=True, copy=False, params=None):
    # Whiten and reduce dimension
    if params:
        pca = params['pca']
        features = pca.transform(features)
    else:
        pca = PCA(n_components=d, whiten=whiten, copy=copy, svd_solver='full')
        features = pca.fit_transform(features)
        params = { 'pca': pca }
    # Normalize
    features = normalize(features, copy=copy)
    return features, params


def query_images():
    query_names = []
    fake_query_names = []
    feats_r = []
    dataset = 'oxford'

    for f in glob.iglob(os.path.join(gt_files, '*_query.txt')):
        fake_query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        fake_query_names.append(fake_query_name)

        query_name, x1, y1, x2, y2 = open(f).read().strip().split(' ')

        if dataset == 'oxford':
            query_name = query_name.replace('oxc1_', '')
            query_names.append('%s.jpg' % query_name)

        ori_cor = tuple((x1, y1, x2, y2))
        final_feat_r = np.zeros((feats_length, ))
        for S in Ss:
            im = cv2.imread(os.path.join(dir_images, '%s.jpg' % query_name))
            ori_shape = im.shape
            im_size_hw = np.array(im.shape[0:2])
            ratio = float(S)/np.max(im_size_hw)
            new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
            if np.min(new_size[0:2]) < 224:
                ratio = float(224)/np.min(im_size_hw)
                new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
            im_resized = cv2.resize(im, (new_size[1], new_size[0]))
            new_shape = im_resized.shape
            new_cor = tuple([round(int(c)*new_shape[0]*1.0/ori_shape[0]) for c in ori_cor])
            I = im_resized.transpose(2, 0, 1) - mean_array

            feats = image_helper.get_ori_features(layers, I, net)
            concated_feature = np.array(())
            for feat in feats:
                feat_cor = tuple([int(round(c*feat.shape[2]*1.0/new_shape[0])) for c in new_cor])
                bbox_regions = np.array([[feat_cor[0], feat_cor[1], feat_cor[2]-feat_cor[0]+1, feat_cor[3]-feat_cor[1]+1]])
                ori_feature = feat[0]
                g, r = image_helper.get_rmac_features_and_final_feature(ori_feature, bbox_regions)
                concated_feature = np.concatenate((concated_feature, r), axis=0)
            final_feat_r += normalize(concated_feature)
        feats_r.append(np.array(final_feat_r))

    return feats_r, query_names, fake_query_names

def query_images_noBoxRmac():
    fake_query_names = []

    query_feats_r = np.zeros((55, feats_length))
    query_names = []
    for i, f in enumerate(glob.iglob(os.path.join(gt_files, '*_query.txt'))):
        fake_query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        fake_query_names.append(fake_query_name)

        query_name, x1, y1, x2, y2 = open(f).read().strip().split(' ')
        query_name = query_name.replace('oxc1_', '') + '.jpg'
        query_names.append(query_name)

        for j in xrange(feats_r.shape[0]):
            if names[j] == query_name:
                print names[j]
                query_feats_r[i] += feats_r[j]

    return query_feats_r, query_names, fake_query_names

def compute_cosin_distance(Q, feats, names):
    # feats and Q: L2-normalize, n*d
    dists = np.dot(Q, feats.T)
    idxs = np.argsort(dists)[::-1]
    rank_dists = dists[idxs]
    rank_names = [names[k] for k in idxs]
    return (idxs, rank_dists, rank_names)

def compute_euclidean_distance(Q, feats, names, k = None):
    if k is None:
        k = len(feats)

    dists = ((Q - feats)**2).sum(axis=1)
    idx = np.argsort(dists) 
    dists = dists[idx]
    rank_names = [names[k] for k in idx]
    return (idx[:k], dists[:k], rank_names)
    
def database_side_feature_aug(Q, data, inds, top_k = 10):
    # weighted query
    for i in range(top_k):
        Q += (1.0*(top_k-i)/float(top_k))*data[inds[i], :]
    return normalize(Q)

def query_expansion(Q, data, inds, top_k = 10):
    Q += data[inds[:top_k], :].sum(axis=0)
    return normalize(Q)

def reranking(Q, data, inds, names, top_k = 50):
    vecs_sum = data[0, :]
    for i in range(1, top_k):
        vecs_sum += data[inds[i], :]
    vec_mean = vecs_sum/float(top_k)
    Q = normalize(Q - vec_mean)
    for i in range(top_k): 
        data[i, :] = normalize(data[i, :] - vec_mean)
    sub_data = data[:top_k]
    sub_idxs, sub_rerank_dists, sub_rerank_names = compute_cosin_distance(Q, sub_data, names[:top_k])
    names[:top_k] = sub_rerank_names
    return names

def load_files(files):
    h5fs = {}
    for i, f in enumerate(files):
        h5fs['h5f_' + str(i)] = h5py.File(f, 'r')
    feats = np.concatenate([value['feats'] for key, value in h5fs.items()])
    names = np.concatenate([value['names'] for key, value in h5fs.items()])
    return (feats, names)

def rank(feats_r, names, 
    query_names, query_feats_r, fake_query_names, 
    gt_files):
    start = timeit.default_timer()
    print feats_r.shape
    feats_r = normalize(feats_r, copy=False)
    if do_pca:
        feats_r, whitening_params_r = run_feature_processing_pipeline(feats_r, d=redud_d, whiten=True, copy=True)
    end = timeit.default_timer()
    print 'feats_r:', feats_r.shape, 'time:', end - start
    aps = []
    rank_file = 'tmp.txt'
    for i, query in enumerate(query_names):
        Q = query_feats_r[i]
        Q = normalize([Q], copy=False)
        if do_pca:
            Q, _ = run_feature_processing_pipeline(Q, params=whitening_params_r)
        Q = np.squeeze(Q.astype(np.float32))
  
        idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats_r, names)
        # sys.exit(0)
        if do_QE:
            Q = query_expansion(Q, feats_r, idxs, top_k = QE_topK)
            idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats_r, names)

        if do_DBA:
            Q = database_side_feature_aug(Q, feats_r, idxs, top_k = DBA_topK)
            idxs, rank_dists, rank_names = compute_cosin_distance(Q, feats_r, names)

        f = open(rank_file, 'w')
        f.writelines([name.split('.jpg')[0] + '\n' for name in rank_names])
        f.close()

        gt_prefix = os.path.join(gt_files, fake_query_names[i])
        # gen result        
        gt_image_names = []
        for l in open(gt_prefix+'_good.txt', 'r'):
            gt_image_names.append(l.strip())
        for l in open(gt_prefix+'_ok.txt', 'r'):
            gt_image_names.append(l.strip())

        fr = open('../res/tmp/%s.txt' % query.split('.jpg')[0], 'w')
        for k in xrange(len(rank_names)):
            flag = 0
            if rank_names[k].split('.jpg')[0] in gt_image_names:
                flag = 1
            fr.write(rank_names[k].split('.jpg')[0] + ' ' + str(rank_dists[k]) + ' ' + str(flag) + '\n')
        fr.close()
        #
        cmd = '../tools/compute_ap %s %s' % (gt_prefix, rank_file)
        ap = os.popen(cmd).read()
        os.remove(rank_file)
        aps.append(float(ap.strip()))
        
        if float(ap.strip()) < 0.7:
            print float(ap.strip()), fake_query_names[i], query
    return np.array(aps).mean()

if __name__ == '__main__':
    # settings ****************************************************
    root_path = "/home/huangxuankun/workspace/ImageRetrieval/cnn-cbir-benchmark"
    gt_files = root_path + '/data/Oxford_Buildings_Dataset/gt_files_170407/'
    dir_images = root_path + '/data/Oxford_Buildings_Dataset/oxbuild_images'

    # feats_files = root_path + '/feats/resnet50/multi_layer_2c_3d_4f_5c/*'
    # mean_array = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]
    # modelDir = "/home/huangxuankun/workspace/ImageRetrieval/cnn-cbir-benchmark/model/resnet50"
    # MODEL =  "resnet-50.caffemodel"
    # PROTO = "sub_deploy.prototxt"
    # caffemodel = os.path.join(modelDir, MODEL)
    # prototxt = os.path.join(modelDir, PROTO)
    # layers = ['res2c', 'res3d', 'res4f', 'res5c']

    feats_files = root_path + '/feats/resnet50/ft_softmax/*'
    mean_array = np.array([104,  117,  123], dtype=np.float32)[None, :, None, None]
    modelDir = "/home/huangxuankun/workspace/ImageRetrieval/ft-resnet50/"
    prototxt = os.path.join(modelDir, 'deploy.prototxt')
    caffemodel = os.path.join(modelDir, 'model/softmax/ft-all-0.01-resnet-50_iter_95000.caffemodel')
    layers = ['res2c', 'res3d', 'res4f', 'res5c']

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    feats_length = 3840
    do_pca = True
    redud_d = 512
    # improvement
    do_QE = False
    QE_topK = 1
    do_DBA = True
    DBA_topK = 20

    S = 800
    multires = True
    Ss = [S, ] if not multires else [S - 250, S, S + 250] # 550 800 1050
    L = 3
    image_helper = ImageHelper(S, L, mean_array)

    # load dataset features *************************************************
    start = timeit.default_timer()
    files =  glob.glob(feats_files)
    feats_r, names = load_files(files)
    stop = timeit.default_timer()
    print "load all features time: %f seconds\n" % (stop - start)

    # get query features ****************************************************
    start = timeit.default_timer()
    query_feats_r, query_names, fake_query_names = query_images()
    print query_feats_r
    print query_names
    stop = timeit.default_timer()
    print 'len(query_names)', len(query_names)
    # comput ap for every query image *********************************************
    mAP = rank(feats_r, names, 
        query_names, query_feats_r, fake_query_names, 
        gt_files)
    print '\nWow mAP is', mAP
