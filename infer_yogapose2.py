import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
from ultralytics import YOLO
from math import ceil
import sys

sys.path.append('/home/uas/trauma/ViTPose')
sys.path.append('/home/uas/trauma/mmcv')

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main(sample_video_path=None,sample_video_folder_path=None,actual_video_dir_path=None):
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path',default='/home/uas/trauma/yogapose/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset(1)/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset/Yoga Postures Dataset/Videos/Anantasana/Anantasana Right Steps/Anantasana Right Step Angle 1-converted.mp4')


    parser.add_argument(
        '--out-video-root',
        default='sample',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Keypoint radius for visualization')

    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()


    yolo_model = YOLO("/home/uas/trauma/ViTPose/demo/best_body.pt")
    pose_model = init_pose_model('/home/uas/trauma/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py', '/home/uas/trauma/ViTPose/demo/vitpose_small.pth', device='cuda:0')
    # print(pose_model)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    return_heatmap = False
    output_layer_names = None
    
    "Trying inference_top_down_pose_model for video"

    """ CASE 1 """

    if sample_video_path is not None:
        cap = cv2.VideoCapture(args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        tracker = None
        bbox = None
        while (cap.isOpened()):
            _,img = cap.read()
            if not _:
                break

            yolo_results = yolo_model(img,stream = True)
            # print("hahaha")

            # person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

            if bbox is None:
                for yr in yolo_results:
                    boxes = yr.boxes
                    for box in boxes:
                        x1,y1,x2,y2 = box.xyxy[0]
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

                        cls = int(box.cls[0])

                        conf = ceil(box.conf[0]*100)/100

                        if cls == 0 and conf > 0.5:
                            bbox = (x1, y1, x2 - x1, y2 - y1)
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(img, bbox)
                            break

            if tracker is not None and bbox is not None:
                success, bbox = tracker.update(img)
                if success:
                    x, y, w, h = map(int, bbox)
                    person_results = [{'bbox': np.array([x, y, x + w, y + h])}]
                else:
                    # Tracker failed, reset
                    bbox = None
                    person_results = [{'bbox': np.array([0, 0, size[0], img.size[1]])}]
            else:
                # No tracker or detection failed, use the full frame
                person_results = [{'bbox': np.array([0, 0, size[0], img.size[1]])}]

            # print(person_results)
            # sys.exit()
            pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

            larm=[pose_results[0]['keypoints'][5][0:2],pose_results[0]['keypoints'][7][0:2],pose_results[0]['keypoints'][9][0:2]]
            rarm=[pose_results[0]['keypoints'][6][0:2],pose_results[0]['keypoints'][8][0:2],pose_results[0]['keypoints'][10][0:2]]
            lleg=[pose_results[0]['keypoints'][11][0:2],pose_results[0]['keypoints'][13][0:2],pose_results[0]['keypoints'][15][0:2]]
            rleg=[pose_results[0]['keypoints'][12][0:2],pose_results[0]['keypoints'][14][0:2],pose_results[0]['keypoints'][16][0:2]]
            keypoints = [larm, rarm, lleg, rleg]
            
            vis_img = vis_pose_result(
                pose_model,
                img,
                pose_results,
                radius=args.radius,
                thickness=args.thickness,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                show=False)
        

            if True:
                show=cv2.resize(vis_img,(900,900))
                cv2.imshow('Image', show)

            if True and cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    sample_video_path = "yogapose/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset(1)/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset/Yoga Postures Dataset/Videos/Ardhakati Chakrasana/Ardhakati Chakrasana Right Steps/Ardhakati Chakrasana Right Step Angle 1-converted.mp4"
    sample_video_folder_path = "yogapose/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset(1)/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset/Yoga Postures Dataset/Videos/Ardhakati Chakrasana"
    actual_video_dir_folder = "yogapose/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset(1)/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset/Yoga Postures Dataset/Videos"
    main(sample_video_path)

