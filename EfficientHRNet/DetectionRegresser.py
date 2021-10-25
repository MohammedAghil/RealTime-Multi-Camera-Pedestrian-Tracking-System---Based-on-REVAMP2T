import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import linear_sum_assignment

# Utility functions copied from EdgeNode:
def xywh2xyxy(bbox):
    bbox = np.array(bbox)
    bbox = bbox.reshape(4)
    x0 = bbox[0]
    x1 = bbox[0] + bbox[2]
    y0 = bbox[1]
    y1 = bbox[1] + bbox[3]

    newbbox = [x0, y0, x1, y1]
    return np.array(newbbox, dtype=int)
    
def xyxy2xywh(bbox):
    bbox = np.array(bbox)
    bbox = bbox.reshape(4)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    return np.array([x,y,w,h], dtype=int)

def IoU(bbox0, bbox1):
    _bbox0 = xywh2xyxy(bbox0)
    x0_1 = _bbox0[0]
    x0_2 = _bbox0[2]
    y0_1 = _bbox0[1]
    y0_2 = _bbox0[3]

    _bbox1 = xywh2xyxy(bbox1)
    x1_1 = _bbox1[0]
    x1_2 = _bbox1[2]
    y1_1 = _bbox1[1]
    y1_2 = _bbox1[3]

    if ((x0_1 > x1_2) or (x0_2 < x1_1) or (y0_1 > y1_2) or (y0_2 < y1_1)):
        return 0
    else:
        dx = min((x0_2, x1_2)) - max((x0_1, x1_1))
        dy = min((y0_2, y1_2)) - max((y0_1, y1_1))
        area = bbox0[2] * bbox0[3] + bbox1[2] * bbox1[3]
        intersection = dx * dy
        union = area - intersection
        print("Union : "+ str(union))
        return intersection/union

# Another utility function
def bboxFromKeypoints(keypoints):
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    
    for xy in keypoints:
        x = xy[0]
        y = xy[1]
        if x != -1 and y != -1:
            if xmin is None or x < xmin:
                xmin = x
            if xmax is None or x > xmax:
                xmax = x
            if ymin is None or y < ymin:
                ymin = y
            if ymax is None or y > ymax:
                ymax = y
    bbox = [xmin, ymin, xmax, ymax]
    bbox = np.array(bbox)
    bbox = xyxy2xywh(bbox)
    return bbox
            

# Just a helpful struct to clean up some code later
class DetectionEntry():
    def __init__(self, timestamp, bbox, keypoints, perc):
        self.t = timestamp
        self.bbox = bbox
        self.keypoints = keypoints
        self.perc = perc
    # def get_t(self):
    #     return self.t
    # def get_bbox(self):
    #     return self.bbox
    # def get_kp(self):
    #     return self.keypoints

class DetectionRegresser():
    def __init__(self, timestamp = 0, window_length = 15, iou_threshold = 0.05,
                 forgetfulness = 2, regress_type = 'corners'):
        self.t = timestamp
        self.window_length = window_length
        self.iou_threshold = iou_threshold
        self.forgetfulness = forgetfulness
        self._regress_type = regress_type # "corners" or "keypoints"
        self.histories = []
        self.models = []
       
        
    # Take in the first set of bboxes and keypoints
    def initialize(self, detections):
        for d in detections:
            self.histories.append([d])
        
        
    # Determine which history best represents the given bbox
    def findBestHistory(self, bbox):
        if not self.histories:
            return None
        
        # Find greatest IoU among previous timestep's detections
        maxiou = -1
        maxh = None
        for h in self.histories:
            iou = IoU(bbox, h[-1].bbox)
            if iou > maxiou:
                maxiou = iou
                maxh = h
        
        # Thresholding
        if maxiou < self.iou_threshold:
            return None
        else:
            return maxh
        
        
    # Print debug info for a given assignment solution
    def assess_assignments(self, d, h, G):
        total_fitness = sum(G[d,h])
        average_fitness = total_fitness / len(d)
        return 'Assignment found with total IoU fitness of {:.3f} ({:.3f} average)'.format(total_fitness, average_fitness)
    
        
    # Solve the problem of matching detections to histories by measuring
    #   IoU and representing it as a linear assignment problem
    def assignment_solver(self, detections):
        # Build array encoding bipartite graph: entries are IoU fitness,
        #   rows are detections, columns are histories
        G = []
        for d in detections:
            match_values = []
            for h in self.histories:
                match_values.append(IoU(d.bbox, h[-1].bbox))
                
            # Add dummy values so new detections have enough "histories" to match with
            # (these will later be assigned a new history, since 0 < threshold)
            for i in range(len(self.histories), len(detections)):
                match_values.append(0)
            
            G.append(match_values)
        
        G = np.array(G)
        
        # Use hungarian algorithm to generate assignments
        assigned_det, assigned_his = linear_sum_assignment(G, maximize=True)
        fitness_debug_str = self.assess_assignments(assigned_det, assigned_his, G)
        
        # For each assignment above threshold, add detection to that history.
        # Otherwise, create a new history from the detection.
        for d_idx, h_idx in zip(assigned_det, assigned_his):
            d = detections[d_idx]
            if G[d_idx, h_idx] > self.iou_threshold:
                self.histories[h_idx].append(d)
            else:
                self.histories.append([d])
        
        if len(self.histories) < len(detections):
            print('DetectionRegresser: Not all detections were assigned to a history!')
        
        return fitness_debug_str
    
    
    # Decide which type of model to build
    def build_model(self, h):
        if self._regress_type == 'keypoints':
            return self.build_model_kp(h)
        elif self._regress_type == 'corners':
            return self.build_model_cn(h)
    
    
    # Build a keypoint LinearRegression model from a history of detections
    def build_model_kp(self, h):
        model = []
        
        # ASSUME: keypoint length stays the same
        # Make one linear regression for each keypoint's coordinates
        for i in range(len(h[0].keypoints)):
            # x and y arrays for regression (x: time, y: coords)
            time = []
            coords = []
            
            # Iterate along timesteps to build arrays
            for detection in h:
                # Ignore outliers / bad values
                xy = detection.keypoints[i]
                if xy[0] != -1 and xy[1] != -1:
                    time.append([detection.t])
                    coords.append(xy)
            
            # Train linear regression model for this one keypoint
            reg = None
            if time and coords:
                reg = LinearRegression()
                reg.fit(time, coords)
            model.append(reg)
        return model
    
    
    # Build a bbox corner LinearRegression model from a history of detections
    def build_model_cn(self, h):
        model = []
        
        # Make one linear regression for the bbox's two corners
        for i in range(2):
            # x and y arrays for regression (x: time, y: coords)
            time = []
            coords = []
            
            # Iterate along timesteps to build arrays
            for detection in h:
                # Convert this bbox to xyxy
                bbox = xywh2xyxy(detection.bbox)
                
                # Grab the corner of this detection (i=0 bottom left, i=1 top right)
                x_coord = bbox[2*i]
                y_coord = bbox[2*i + 1]
                xy = [x_coord, y_coord]
                
                time.append([detection.t])
                coords.append(xy)
            
            # Train linear regression model for this one corner
            reg = None
            if time and coords:
                reg = LinearRegression()
                reg.fit(time, coords)
            model.append(reg)
        return model
        
    
    # Take in a new set of bboxes and keypoints, determine which history they
    # belong to, and update models
    def update(self, bboxes, keypoints, percentages):
        self.t += 1
        
        # Generate detections
        detections = []
        for bb, kp, perc in zip(bboxes, keypoints, percentages):
            detections.append(DetectionEntry(self.t, bb, kp, perc))
        
        # Initialize if this is the first run
        if not self.histories:
            self.initialize(detections)
            debug_str = 'Initializing DetectionRegresser'
        else:
            debug_str = self.assignment_solver(detections)
        
        # Remove histories that have not been recently updated
        cutoff = self.t - self.forgetfulness
        self.histories = [h for h in self.histories if h[-1].t >= cutoff]
        
        # Build models
        self.models = []
        for h in self.histories:
            self.models.append(self.build_model(h))
            
        return debug_str
        
        
    # Decide which type of prediction to do
    def predict(self):
        if self._regress_type == 'keypoints':
            return self.predict_kp()
        elif self._regress_type == 'corners':
            return self.predict_cn()
            
        
    # Uses stored keypoint models to predict a given timestamp
    def predict_kp(self, timestamp = -1):
        if not self.models:
            return None
        if timestamp == -1:
            timestamp = self.t
            
        predicted_bboxes = []
        predicted_keypoints = []
        latest_percentages = []
        
        # Iterate over detections' models
        for detection_model, h in zip(self.models, self.histories):
            # TODO: how should we decay the 'confidence' of old detections?
            if(h[-1].t < self.t):
                latest_percentages.append(0)
            else:
                latest_percentages.append(h[-1].perc)
            
            # Iterate over keypoints' models
            keypoints = []
            for i, m in enumerate(detection_model):
                # If model doesn't exist, just use last detection
                if m is None:
                    keypoints.append(h[-1].keypoints[i])
                else:
                    # Predict and add
                    # (parens and brackets are really odd here due to dimensionality
                    #  of the LinearRegression class' inputs/outputs)
                    keypoints.append(m.predict([[timestamp]])[0])
            predicted_keypoints.append(keypoints)
            predicted_bboxes.append(bboxFromKeypoints(keypoints))
        
        return predicted_bboxes, predicted_keypoints, latest_percentages
            
        
    # Uses stored corner models to predict a given timestamp
    def predict_cn(self, timestamp = -1):
        if not self.models:
            return None
        if timestamp == -1:
            timestamp = self.t
            
        predicted_bboxes = []
        predicted_keypoints = []
        latest_percentages = []
        
        # Iterate over detections' models
        for detection_model, h in zip(self.models, self.histories):
            # TODO: how should we decay the 'confidence' of old detections?
            if(h[-1].t < self.t):
                latest_percentages.append(0)
            else:
                latest_percentages.append(h[-1].perc)
            
            # Iterate over corners' models
            corners = []
            for i, m in enumerate(detection_model):
                # If model doesn't exist, just use last detection
                if m is None:
                    corners.append([h[-1].bbox[2*i], h[-1].bbox[2*i + 1]])
                else:
                    # Predict and add
                    # (parens and brackets are really odd here due to dimensionality
                    #  of the LinearRegression class' inputs/outputs)
                    corners.append(m.predict([[timestamp]])[0])
                    
            predicted_keypoints.append(h[-1].keypoints) # Don't predict, use latest
            predicted_bboxes.append(bboxFromKeypoints(corners)) # Easier to use here than xyxy2xywh
        
        return predicted_bboxes, predicted_keypoints, latest_percentages