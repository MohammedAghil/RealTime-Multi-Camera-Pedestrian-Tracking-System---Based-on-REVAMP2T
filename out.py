import os
import cv2

from EfficientHRNet.EfficientHRNetKeypointExtractor import EfficientHRNetKeypoints
from EfficientHRNet.lib.config import pose_cfg, update_config, check_config
from Helpers.pre_process import *
from Helpers import options
from ReIDFeatureExtractor.ReIDFeatureEncoder import FeatureEncoder
from DeepSort.deep_sort_main import DeepSort

if __name__ == '__main__':
    opts = options.parse_args()
    # Open master YAML config
    master_dict = open_yaml_file(opts.yaml_master)
    # print(opts.pose_cfg)
    update_config(pose_cfg, master_dict['pose-config'])
    check_config(pose_cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(master_dict['use-gpu'])
    use_gpu = (master_dict['use-gpu'] != -1)

    # Open YAML file for EfficientHRNET
    nodes_dict = open_yaml_file(master_dict['node-config'])
    defaults_dict = nodes_dict['default']
    nodes_dict = dict(nodes_dict['nodes'])

    # Open YAML files for plrosnet and deepsort
    plrosnet_dict,deepsort_dict = open_multiple_yaml_files([master_dict['plrosnet'],master_dict['deepsort']])

    # 1) EfficientHRNet Object
    EfficientHRNet = EfficientHRNetKeypoints(pose_cfg, defaults_dict)

    # 2) PLROSNET Object
    print(master_dict['plrosnet'])
    PLROSNET_Obj = FeatureEncoder(plrosnet_dict, use_gpu)

    # 3) DeepSort Object
    DeepSort_Obj = DeepSort(deepsort_dict)
    image_path, image_output_path, sequence = pre_process(master_dict, nodes_dict)

    print("\nStarting Execution :-")
    for frame in range(sequence[0], sequence[1]):
        image = cv2.imread(image_path + '/' + "{:04}".format(int(frame)) + '.jpg', None)
        if image is not None:
            print("\nProcessing Frame : {0:4d}".format(int(frame)))
            # 1) TODO : EfficientHRNet
            bboxes, final_keypoints, valid_keypoints, img, image_crops_list = EfficientHRNet.run(frame, image,
                                                                                                 image_path,
                                                                                                 image_output_path)
            # 2) TODO : PLR OSNET
            encodedFeatures = PLROSNET_Obj.run(frame, image, image_output_path, bboxes, valid_keypoints)

            print("BBOXES : {} ,  PLROSNET Features :{} , Feature Vector Size : {}".format(len(bboxes),
                                                                                           len(encodedFeatures),
                                                                                           len(encodedFeatures[0])))
            # we have no of people = no of bounding boxes and the encodedfeatures
            # print(len(valid_keypoints))
            # 3) TODO : Deep Sort
            # bboxes + encodedFeatures + valid_keypoints
            DeepSort_Obj.run(image=image, frame=frame, bboxes=bboxes, features=encodedFeatures,
                             confidence=valid_keypoints, image_output_path=image_output_path)
            # print(len(deep_sort_results), len(bboxes))
        else:
            print("No Image Found {:04}".format(frame))
