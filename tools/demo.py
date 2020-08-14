from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

import time

'''
python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth  \
    --video demo/bag.avi  \
    --writeout True

python tools/demo.py \
    --config experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth  \
    --video demo/bag.avi  \
    --writeout True

python tools/demo.py \
    --config experiments/siamrpn_alex_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_alex_dwxcorr/model.pth  \
    --video demo/bag.avi  \
    --writeout True

python tools/demo.py \
    --config experiments/siammask_r50_l3/config.yaml \
    --snapshot experiments/siammask_r50_l3/model.pth  \
    --video demo/bag.avi  \
    --writeout True

python tools/demo.py \
    --config experiments/siamcar_r50/config.yaml \
    --snapshot experiments/siamcar_r50/model.pth  \
    --video demo/bag.avi  \
    --writeout True

'''



torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--writeout', default=False,type=bool,
                    help='write to a file if True')
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
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
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

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=device))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    img_array=[]
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
                print('bounding box is captured\n', init_rect)
            except:
                exit()
            init_start = time.time()
            tracker.init(frame, init_rect)
            #draw the user input bounding box
            cv2.rectangle(frame, (init_rect[0], init_rect[1]), 
                (init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]), 
                (0,255,0), 3)
            img_array.append(frame)
            first_frame = False
        else:
            print('processing frame')
            start = time.time()
            outputs = tracker.track(frame)
            print('Frame processing time is {} second'.format(round(time.time() - start, 4)))
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
            if args.writeout:
                img_array.append(frame)
            else:
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)

            if len(img_array)>1:
                height, width, _ = img_array[0].shape
                size = (width, height)
                file_name = 'output' + video_name
                out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
        print('Accumulated processing time for this video is {} seconds'.format(round(time.time() - init_start,4)))

if __name__ == '__main__':
    main()
