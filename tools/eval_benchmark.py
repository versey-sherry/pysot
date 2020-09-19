from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('./')

import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
#all trackers from pysot
from pysot.tracker.tracker_builder import build_tracker
#adding SiamCAR Tracker
from pysot.tracker.siamcar_tracker import SiamCARTracker
#SiamCAR uses a older type model snapshot
from pysot.utils.model_load import load_pretrain

import time


'''
python tools/eval_benchmark.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth  \
    --video_folder LaSOT \
    --gt groundtruth.txt

python tools/eval_benchmark.py \
    --config experiments/siammask_r50_l3/config.yaml \
    --snapshot experiments/siammask_r50_l3/model.pth  \
    --video_folder data/LS \
    --gt groundtruth.txt \
'''


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_folder', default='', type=str,
                    help='folder that contains all the videos')
parser.add_argument('--bbox', default='', type=str, help='bounding box for consistent user input init box')
parser.add_argument('--gt', default='', type=str, help='bounding box ground truth name type')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        print('reading image')
        print(os.path.join(video_name, '*/*.jp*'))
        images = glob(os.path.join(video_name, '*/*.jp*'))
        #print(images)
        #control the length of images to control the number of frames
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print('Current device is', device)
    print('current eval is', args.video_folder)

    # create model
    model = ModelBuilder()

    if 'siamcar' in args.config:
        model = load_pretrain(model, args.snapshot).eval().to(device)
        model.eval().to(device)
        #build tracker
        tracker = SiamCARTracker(model, cfg.TRACK)
        hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}
    else:
        # load model
        model.load_state_dict(torch.load(args.snapshot,
            map_location=device))
        model.eval().to(device)

        # build tracker
        tracker = build_tracker(model)

    #process initial bounding box
    if args.bbox:
        print(args.bbox)
        args.bbox = eval(args.bbox)

    #list the name of all the videos
    video_list = [file for file in os.listdir(args.video_folder) if not file.startswith('.')]
    print(video_list)
    time_list = []
   
    a = 0
    for a in range(len(video_list)):
        print(video_list[a])
        current_dir = video_list[a]
    
        # load ground truth
        bbox_gt = []
        if args.gt:
            gt_file = os.path.join(args.video_folder, current_dir, args.gt)
            print('ground truth file is', gt_file)
            with open(gt_file) as file:
                for line in file:
                    try:
                        bbox_gt.append(eval(line))
                    except:
                        line = [int(item) for item in line.split()]
                        bbox_gt.append(line)
            #use ground truth bbox as the init box
            args.bbox = bbox_gt[0]  
            print(args.bbox) 

        first_frame = True
        video_start = time.time()
        result_list = []
        b = 0
        
        for frame in get_frames(os.path.join(args.video_folder, current_dir)):
            b+=1
            if first_frame:
                if args.bbox:
                    init_rect = args.bbox
                else:
                    raise ValueError

                tracker.init(frame, init_rect)
                first_frame = False
            else:
                print('processing frame', b)
                if 'siamcar' in args.config:
                    outputs=tracker.track(frame, hp)
                else:
                    outputs = tracker.track(frame)
                #output bbox is x, y, h, w format
                print([b, outputs['bbox'][0], outputs['bbox'][1], outputs['bbox'][2],outputs['bbox'][3], outputs['best_score']])
                result_list.append([b, outputs['bbox'][0], outputs['bbox'][1], outputs['bbox'][2],outputs['bbox'][3], outputs['best_score']])

        if not os.path.exists('results'):
            os.makedirs('results')

        output_file = os.path.join('results', args.config.split('/')[-2]+current_dir+'_result.txt')

        with open(output_file, 'w') as file:
            for det in result_list:
                det = '{}, {}, {}, {}, {}'.format(det[1], det[2], det[3], det[4], det[5])
                print(det)
                file.write('%s\n' % str(det))
        print('total consumed time is', time.time()-video_start, 'second')
        print(output_file, 'is done')


if __name__ == '__main__':
    main()
