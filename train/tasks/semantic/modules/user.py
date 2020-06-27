#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import os
import time

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch import nn

import __init__ as booger
from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN

def get_sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, modelname, split):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.modelname = modelname
        self.split = split

        # get the data
        parserModule = imp.load_source('parserModule',
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' + self.DATA['name'] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA['split']['train'],
                                          valid_sequences=self.DATA['split']['valid'],
                                          test_sequences=self.DATA['split']['test'],
                                          labels=self.DATA['labels'],
                                          color_map=self.DATA['color_map'],
                                          learning_map=self.DATA['learning_map'],
                                          learning_map_inv=self.DATA['learning_map_inv'],
                                          sensor=self.ARCH['dataset']['sensor'],
                                          max_points=self.ARCH['dataset']['max_points'],
                                          batch_size=1,
                                          # workers=1,
                                          # important for time measurement
                                          workers=0,
                                          gt=True,
                                          shuffle_train=False)

        # concatenate the encoder and the head
        if self.modelname in ('salsanet', 'salsanext'):
            with torch.no_grad():
                print('modeldir: %s' % self.modeldir)
                model_path = os.path.join(self.modeldir, 'SalsaNet')
                print('model_path: %s' % model_path)

                self.model = SalsaNet(self.ARCH,
                                      self.parser.get_n_classes(),
                                      model_path)
                self.model = nn.DataParallel(self.model)
                torch.nn.Module.dump_patches = True

                w_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
                print(w_dict['state_dict'].keys())

                self.model.module.load_state_dict(w_dict['state_dict'], strict=True)
        else:
            with torch.no_grad():
                self.model = Segmentator(self.ARCH,
                                         self.parser.get_n_classes(),
                                         self.modeldir)

        # use knn post processing?
        self.post = None
        if self.ARCH['post']['KNN']['use']:
            self.post = KNN(self.ARCH['post']['KNN']['params'], self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Infering in device: ', self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()


    def infer(self):
        if self.split == None:
            # do train set
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original)
            # do valid set
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original)
            # do test set
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original)
        elif self.split == 'valid':
            self.infer_subset(loader=self.parser.get_valid_set(),
                              to_orig_fn=self.parser.to_original)
        elif self.split == 'train':
            self.infer_subset(loader=self.parser.get_train_set(),
                              to_orig_fn=self.parser.to_original)
        else:
            self.infer_subset(loader=self.parser.get_test_set(),
                              to_orig_fn=self.parser.to_original)

        print('Finished Infering')

        return

    def infer_subset(self, loader, to_orig_fn):
        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            # infer time segments
            infer_times = []
            # projection time segments
            proj_times = []
            
            for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints, proj_time) in enumerate(loader):
                proj_times.append(proj_time.data.cpu().numpy()[0])

                # first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                # loading data on GPU
                # depends on GPU, so we dont include this in the inference time
                if self.gpu:
                    proj_in = proj_in.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                # INFER TIME START
                infer_time_start = get_sync_time()

                # compute output
                proj_output = self.model(proj_in)
                proj_argmax = proj_output[0].argmax(dim=0)

                if self.post:
                    # knn postproc
                    unproj_argmax = self.post(proj_range, unproj_range, proj_argmax, p_x, p_y)
                else:
                    # put in original pointcloud using indexes
                    unproj_argmax = proj_argmax[p_y, p_x]

                # INFER TIME END
                infer_time_end = get_sync_time()

                infer_times.append(infer_time_end - infer_time_start)
                print('Infered sequence: %s' % path_seq)
                print('Scan: %s' % path_name)
                print('Proj time: %s sec' % proj_times[-1])
                print('Infer time: %s sec' % infer_times[-1])
                print('Total time: %s sec' % (proj_times[-1] + infer_times[-1]))

                # save scan
                # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                # save scan
                path = os.path.join(self.logdir, 'sequences', path_seq, 'predictions', path_name)
                pred_np.tofile(path)
                

            print('*' * 30)
            print('INFER TIME STATISTICS')
            print('MEAN: %s' % np.mean(infer_times[1:]))
            print('STD: %s' % np.std(infer_times[1:]))
            print('COUNT: %s' % len(infer_times[1:]))
            # plt.plot(infer_times[1:])
            # plt.savefig('infer_time.png')
            print('-' * 15)
            print('PROJ TIME STATISTICS')
            print('MEAN: %s' % np.mean(proj_times[1:]))
            print('STD: %s' % np.std(proj_times[1:]))
            print('COUNT: %s' % len(proj_times[1:]))
            # plt.plot(proj_times[1:])
            # plt.savefig('proj_time.png')

    def predict(self):
        pass
     

