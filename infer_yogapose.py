# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
from ultralytics import YOLO

# yolo_model = YOLO("ViTPose/demo/best_body.pt")


import sys
sys.path.append('/home/uas/trauma/ViTPose')
sys.path.append('/home/uas/trauma/mmcv')

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('pose_config', help='Config file for pose')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file for pose',default='./home/gunmay/VitPose-s_RePoGen.pth')
    parser.add_argument('--video-path', type=str, help='Video path',default='/home/uas/trauma/yogapose/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset(1)/Yoga for all A Comprehensive Collection of Yoga Images and Videos dataset/Yoga Postures Dataset/Videos/Anantasana/Anantasana Right Steps/Anantasana Right Step Angle 1-converted.mp4')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='sample',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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

    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file
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

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    i=0

    tracker = None
    bbox = None

    while (cap.isOpened()):
        flag, img = cap.read()

        # img=cv2.resize(img,(256,192))
        if not flag:
            break

        # keep the person class bounding boxes.

        # yolo_result = yolo_model.track(img,tracker='sort')
        # yolo_boxes = yolo_result[0].boxes 



        person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

        # for box in yolo_boxes:
        #     class_id = box.cls.cpu().numpy()
        #     if class_id == 0:  
        #         x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        #         yolo_results = [{'bbox': np.array([x1, y1, x2, y2])}]

        # print(yolo_results)

        # continue


        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        # print(pose_results[0]['keypoints'])
        larm=[pose_results[0]['keypoints'][5][0:2],pose_results[0]['keypoints'][7][0:2],pose_results[0]['keypoints'][9][0:2]]
        rarm=[pose_results[0]['keypoints'][6][0:2],pose_results[0]['keypoints'][8][0:2],pose_results[0]['keypoints'][10][0:2]]
        lleg=[pose_results[0]['keypoints'][11][0:2],pose_results[0]['keypoints'][13][0:2],pose_results[0]['keypoints'][15][0:2]]
        rleg=[pose_results[0]['keypoints'][12][0:2],pose_results[0]['keypoints'][14][0:2],pose_results[0]['keypoints'][16][0:2]]
        keypoints = [larm, rarm, lleg, rleg]


        # show the results
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

        if save_out_video:
            videoWriter.write(vis_img)

        if True and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
