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
python tools/demo.py \
    --config experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth  \
    --video data/OTB/Subway/img  \
    --gt data/OTB/Subway/groundtruth_rect.txt  \
    --writeout True

python tools/demo.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth  \
    --video data/OTB/Subway/img  \
    --gt data/OTB/Subway/groundtruth_rect.txt  \
    --writeout True

python tools/demo.py \
    --config experiments/siammask_r50_l3/config.yaml \
    --snapshot experiments/siammask_r50_l3/model.pth  \
    --video data/OTB/Subway/img  \
    --gt data/OTB/Subway/groundtruth_rect.txt  \
    --writeout True

python tools/demo.py \
    --config experiments/siamcar_r50/config.yaml \
    --snapshot experiments/siamcar_r50/model.pth  \
    --video data/OTB/Subway/img  \
    --gt data/OTB/Subway/groundtruth_rect.txt  \
    --writeout True

'''



torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--bbox', default='', type=str, help='bounding box for consistent user input init box')
parser.add_argument('--gt', default='', type=str, help='bounding box ground truth')
parser.add_argument('--writeout', default=False, type=bool,
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
        print(os.path.join(video_name, '*.jp*'))
        images = glob(os.path.join(video_name, '*.jp*'))
        #print(images)
        #control the length of images to control the number of frames
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


# compute IoU between prediction and ground truth, bbox format x1,y1,x2,y2
def compute_iou(prediction, gt):
    #ensure the bounding boxes exist
    assert(prediction[0] <= prediction[2])
    assert(prediction[1] <= prediction[3])
    assert(gt[0] <= gt[2])
    assert(gt[1] <= gt[3])

    #intersection rectangule
    xA = max(prediction[0], gt[0])
    yA = max(prediction[1], gt[1])
    xB = min(prediction[2], gt[2])
    yB = min(prediction[3], gt[3])

    #compute area of intersection
    interArea = max(0, xB-xA + 1) * max(0, yB - yA + 1)

    #compute the area of the prection and gt
    predictionArea = (prediction[2] - prediction[0] +1) * (prediction[3] - prediction[1] +1)
    gtArea = (gt[2] - gt[0] + 1) * (gt[3]-gt[1]+1)

    #compute intersection over union
    iou = interArea / float(predictionArea+gtArea-interArea)
    return iou

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    print('Current device is', device)

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
        args.bbox = eval(args.bbox)

    # load ground truth
    bbox_gt = []
    if args.gt:
        print('ground truth file is', args.gt)
        with open(args.gt) as file:
            for line in file:
                bbox_gt.append(eval(line))
        #use ground truth bbox as the init box
        args.bbox = bbox_gt[0]   


    # process image
    img_array=[]
    iou_array = []
    first_frame = True

    if args.video_name:
        video_name = args.video_name.split('/')[-2].split('.')[0]
        print(video_name)
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    a = 0
    for frame in get_frames(args.video_name):
        if first_frame:
            if args.bbox:
                init_rect=args.bbox
            else:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    print('bounding box is captured\n', init_rect)
                except:
                    exit()
            
            init_start = time.time()
            tracker.init(frame, init_rect)
            #draw init box with white
            cv2.rectangle(frame, (init_rect[0], init_rect[1]), 
                (init_rect[0]+init_rect[2], init_rect[1]+init_rect[3]), 
                (255,255,255), 3)
            img_array.append(frame)
            first_frame = False
        else:
            print('processing frame')
            start = time.time()

            if 'siamcar' in args.config:
                outputs=tracker.track(frame, hp)
            else:
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
                bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                #draw predicted bounding box with red
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              (0, 0, 255), 3)
                #draw gt bounding box with red
                if len(bbox_gt)>0:
                    gt = bbox_gt[a]
                    gt = [gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]]
                    cv2.rectangle(frame, (gt[0], gt[1]), 
                        (gt[2], gt[3]), 
                        (255,255,255), 3)
                    text = 'Frame {}: IoU is {}%'.format(a+1, round((compute_iou(bbox, gt) *100),2))
                    cv2.putText(frame, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA) 
                    iou_array.append(compute_iou(bbox, gt))
                    #print stats
                    print(text)
                    a +=1
            if args.writeout:
                img_array.append(frame)
            else:
                cv2.imshow(video_name, frame)
                cv2.waitKey(1)
    
    print('Mean IoU is', round(sum(iou_array)/len(iou_array) *100, 2), '%')
    print('Total processing time for this video is {} seconds'.format(round(time.time() - init_start,4)))

    if len(img_array)>1:
        height, width, _ = img_array[0].shape
        size = (width, height)
        file_name = '_'.join(['output', video_name, args.config.split('/')[-2]])+'.avi'
        out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


if __name__ == '__main__':
    main()
