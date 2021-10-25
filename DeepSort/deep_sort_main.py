from __future__ import division, print_function, absolute_import

import cv2
import numpy as np
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection
from .application_util import preprocessing
from .deep_sort import nn_matching

class DeepSort:
    def __init__(self,deepsort_dict):
        self.deepsort_dict = deepsort_dict
        self.max_cosine_distance = deepsort_dict['max_cosine_distance']
        self.nn_budget = deepsort_dict['nn_budget']
        self.nms_max_overlap = deepsort_dict['nms_max_overlap']
        self.min_confidence = deepsort_dict['min_confidence']
        self.min_height = deepsort_dict['min_detection_height']
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric,self.deepsort_dict['max_iou_distance'],self.deepsort_dict['max_age'],self.deepsort_dict['n_init'])
        self.ran = 0
        self.results = []

    def create_detections(self,features,bbox,confidence):
        """Create detections for given frame index from the raw detection matrix."""
        detection_list = []
        for i in range(len(bbox)):
            if bbox[i][3] < self.min_height:
                continue
            detection_list.append(Detection(bbox[i], confidence[i], features[i]))
        return detection_list

    def draw_bboxes_and_id(self,frame,image,image_output_path,deep_sort_results):
        for i in range(len(deep_sort_results)):
            if deep_sort_results[i][2] and deep_sort_results[i][3] and deep_sort_results[i][4] and deep_sort_results[i][5]:
                x, y, w, h = int(deep_sort_results[i][2]), int(deep_sort_results[i][3]), int(
                    deep_sort_results[i][4]), int(deep_sort_results[i][5])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, str(deep_sort_results[i][1]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                cv2.imwrite(image_output_path['DeepSort'] + '/' + "{0:4d}".format(int(frame)) + '.jpg',
                            image)



    def run(self,frame,image,bboxes,features, confidence,image_output_path):
        print("DeepSort :- Processing Frame : {0:4d}".format(int(frame)))

        detections = self.create_detections(features=features, bbox=bboxes, confidence=confidence)
        detections = [d for d in detections if d.confidence >= self.min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)
        # Store results.
        for track in self.tracker.tracks:
            print('Track ID :{} -> Time : {} , State : {}'.format(track.track_id, track.time_since_update,track.state))
            if not track.is_confirmed() or (track.time_since_update > 1):
                continue
                #pass
            self.ran+=1
            bbox = track.to_tlwh()
            self.results.append([
                frame, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        print('Ran {} times'.format(self.ran))
        self.draw_bboxes_and_id(frame,image,image_output_path,self.results)
