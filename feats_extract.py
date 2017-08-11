import sys
import numpy as np
import caffe
import cv2
import os
import glob
import h5py
from extract_features import opencv_format_img_for_vgg, extract_fc_features
import multiprocessing
from multiprocessing import Process, freeze_support, Pool
from crow import compute_crow_spatial_weight, compute_crow_channel_weight
from sklearn.preprocessing import normalize as sknormalize

class ImageHelper:
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means
    
    def prepare_image_and_grid_regions_for_network(self, I, im_resized):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            #print self.L         
            #print all_regions 
            R = self.pack_regions_for_network(all_regions)
            #print R.shape
        return I, R

    def get_ori_features(self, layers, I, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.forward()
        ml_data = []
        for layer in layers:
            ml_data.append(net.blobs[layer].data)
        return ml_data

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        if np.min(new_size[0:2]) < 224:
            ratio = float(224)/np.min(im_size_hw)
            new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        print 'regions num', n_regs
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                # r.shape (21, 4)
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)

        return np.array(regions_xywh).astype(np.int)

    def get_rmac_features_and_final_feature(self, ori_feature, all_regions):
        # ori_feature (2048, 13, 18)
        # all_region (21, 4) in (X, Y, W, H) format, eg,.(0, 0, 18, 13)
        vecs_rmac = np.zeros((ori_feature.shape[0], ))
        for i in xrange(all_regions.shape[0]):
            sub_feature = ori_feature[:, all_regions[i][1]:all_regions[i][1]+all_regions[i][3], all_regions[i][0]:all_regions[i][0]+all_regions[i][2]]
            vec_tmp = np.max(sub_feature, axis=(1, 2))
            vecs_rmac += normalize(vec_tmp)
        # global feature
        vecs_global = np.max(ori_feature, axis=(1, 2))
        return vecs_global, vecs_rmac

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

def normalize(x, copy=False):
    if type(x) == np.ndarray and len(x.shape) == 1:
        return np.squeeze(sknormalize(x.reshape(1,-1), copy=copy))
    else:
        return sknormalize(x, copy=copy)

def multi_gpu_task(proto, model, path_images, feature_out_file, gpu_id):
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(proto, model, caffe.TEST)

    features_rmac = []
    image_names = []
    for i, path in enumerate(path_images):
        print i, path
        feature_r_i = np.zeros((feature_length, ))
        for S in Ss:
            image_helper.S = S
            I, im_resized = image_helper.load_and_prepare_image(path, roi=None) # resized ori-image

            ml_ori_feature = image_helper.get_ori_features(layers, I, net)
            concated_feature = np.array(())
            for ori_feature in ml_ori_feature:
                print ori_feature.shape
                ori_feature = ori_feature[0]
                all_regions = image_helper.get_rmac_region_coordinates(ori_feature.shape[1], ori_feature.shape[2], L)
                g, r = image_helper.get_rmac_features_and_final_feature(ori_feature, all_regions)
                concated_feature = np.concatenate((concated_feature, r), axis=0)
            print concated_feature.shape
            feature_r_i += normalize(concated_feature)
        features_rmac.append(feature_r_i)
        image_names.append(os.path.basename(path))
    
    print np.array(features_rmac).shape
    h5f = h5py.File(feature_out_file, 'w')
    h5f['feats'] = np.array(features_rmac)
    h5f['names'] = np.array(image_names)
    h5f.close()
    
if __name__ == '__main__':
    S = 800
    multires = True
    Ss = [S, ] if not multires else [S - 250, S, S + 250] # 550 800 1050
    L = 3
    layers = ['res2c', 'res3d', 'res4f', 'res5c'] # 256 512 1024 2048
    feature_length = 3840

    gpusID = [0, 1, 2, 3, 4, 5, 6, 7]
    parts = len(gpusID)

    feature_out_prefix = '../feats/resnet50/ft_softmax/'

    # means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]
    # modelDir = "/home/huangxuankun/workspace/ImageRetrieval/cnn-cbir-benchmark/model/resnet50"
    # proto = os.path.join(modelDir, 'sub_deploy.prototxt')
    # model = os.path.join(modelDir, 'resnet-50.caffemodel')
    # image_helper = ImageHelper(S, L, means)

    # means = np.array([121.169, 126.03, 128.329], dtype=np.float32)[None, :, None, None]
    means = np.array([104,  117,  123], dtype=np.float32)[None, :, None, None]
    modelDir = "/home/huangxuankun/workspace/ImageRetrieval/ft-resnet50"
    proto = os.path.join(modelDir, 'deploy.prototxt')
    model = os.path.join(modelDir, 'model/softmax/ft-all-0.01-resnet-50_iter_95000.caffemodel')
    image_helper = ImageHelper(S, L, means)

    dir_images = '/home/huangxuankun/workspace/ImageRetrieval/cnn-cbir-benchmark/data/Oxford_Buildings_Dataset/oxbuild_images/*'
    path_images = [os.path.join(dir_images, f) for f in sorted(glob.glob(dir_images)) if f.endswith('.jpg')]
    blocks = split_list(path_images, wanted_parts = parts)

    out_files = []
    for i in xrange(parts):
        out_files.append(os.path.join(feature_out_prefix, 'resnet50_' + str(i) + '.h5'))
    
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    for i in xrange(parts):
        pool.apply_async(multi_gpu_task, args=(proto, model, blocks[i], out_files[i], gpusID[i], ))
    pool.close()
    pool.join()
    print "task has finished! "
