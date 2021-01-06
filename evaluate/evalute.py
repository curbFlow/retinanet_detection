import csv
import numpy as np
import cv2
import pandas as pd
import tensorflow.keras as keras
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image
from tqdm import tqdm
import os


def bb_IOU(box_a, box_b):
    """
    Params:
        box_a: box of the form x1,y1,x2,y2
        box_b: box of the form x1,y1,x2,y2
    Returns:
        iou between box_a and box_b
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class ImageFrame:
    def __init__(self, frame_name):
        self.frame_name = frame_name
        self.bb_true = []
        self.bb_pred = []

    def get_bb_true(self, db):
        """
        Get all the true bounding boxes of the image from a database that has the image.
        """
        df = db[db['Frame'] == self.frame_name]
        if len(df) == 0:
            return
        else:
            self.bb_true = []
            for row in df.iterrows():
                self.bb_true.append(
                    [row[1]['x1'], row[1]['y1'], row[1]['x2'], row[1]['y2'], row[1]['Label']])
            self.bb_true = np.array(self.bb_true)
            return

    def get_bb_pred(self, predictor):
        """
        Get all the predicted bounding boxes of the image using an object detector.
        """
        img = cv2.imread(self.frame_name)
        self.bb_pred = predictor(img)
        return

    def annotate(self, true_bb=True, pred_bb=True):
        """
        Returns an annotated image, with Green boxes on true bb and Red boxes on predicted bb.
        Annotate must be called only after calling get_bb_pred and get_bb_true are called.
        Params:
            true_bb: Whether to annotate true bounding boxes on the image.
            pred_bb: Whether to annotate predicted bounding boxes on the image.
        """
        img = cv2.imread(self.frame_name)
        annotated = np.copy(img)
        if (true_bb):
            try:
                for bb in self.bb_true:
                    cv2.rectangle(annotated, (bb[0], bb[1]),
                                  (bb[2], bb[3]), (0, 255, 0), 3)
            except:
                pass
        if (pred_bb):
            try:
                for bb in self.bb_pred:
                    cv2.rectangle(annotated, (bb[0], bb[1]),
                                  (bb[2], bb[3]), (0, 0, 255), 3)
            except:
                pass

        return annotated

    def score(self, class_id=0, threshold=0.5):
        """
        return TP, FP, FN for a class.
        """
        bb_true = self.bb_true
        bb_pred = self.bb_pred
        try:
            bb_true_class = bb_true[bb_true[:, 4] == class_id]
            bb_pred_class = bb_pred[bb_pred[:, 4] == class_id]
            if (len(bb_true_class) == 0) and (len(bb_pred_class) == 0):
                return 0, 0, 0
            elif (len(bb_true_class) == 0) and (len(bb_pred_class) > 0):
                return 0, len(bb_pred_class), 0
            elif (len(bb_true_class) > 0) and (len(bb_pred_class) == 0):
                return 0, 0, len(bb_true_class)
            else:
                T_match = np.zeros(len(bb_true_class))
                P_match = np.zeros(len(bb_pred_class))
                for idx_t, tb in enumerate(bb_true_class):
                    for idx_p, pb in enumerate(bb_pred_class):
                        if P_match[idx_p]:
                            continue
                        else:
                            iou = bb_IOU(tb, pb)
                            if iou > threshold:
                                T_match[idx_t] = 1
                                P_match[idx_p] = 1
                                continue
                TP = len(T_match[T_match == 1])
                FP = len(P_match[P_match == 0])
                FN = len(T_match[T_match == 0])
                return TP, FP, FN
        except TypeError:
            # bb_pred is none:
            return 0, 0, len(bb_true_class)
        except IndexError:
            # bb_pred is np.array([]):
            return 0, 0, len(bb_true_class)


class RetinanetEval:

    def __init__(self, csv_path, model_path, classes_csv, score_thresh=0.5, scale=1.0, bbox_thresh=0.5):

        self.score_thresh = score_thresh
        self.bbox_thresh = bbox_thresh
        self.model_path = model_path
        self.scale = scale

        with open(classes_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.class_map = {labelname: int(idx) for labelname, idx in reader}

        self.db = pd.read_csv(csv_path, names=['Frame', 'x1', 'y1', 'x2', 'y2', 'Label'])
        self.db['Label'] = self.db['Label'].apply(lambda x: self.class_map[x])
        self.db = self.db[self.db['Label'] >= 0]

        self.detector = ResnetDetector(model_path=self.model_path, score_thresh=self.score_thresh, scale=self.scale)
        self.frames = []
        self.detections_done = False

    def build_index(self):
        """
        Get all image filenames and the true bounding boxes of each.
        """
        self.frames = []
        for fname in self.db['Frame'].unique():
            frame = ImageFrame(fname)
            frame.get_bb_true(self.db)
            self.frames.append(frame)

    def run_detection(self):
        """
        Prepare frames and the true bboxes and get all the predicted bboxes for images in the db.
        """
        if not self.frames:
            self.build_index()
        for frame in tqdm(self.frames):
            frame.get_bb_pred(self.detector.detect)

        self.detections_done = True

    def evaluate_class(self, class_id, labelname):
        """
        Returns precision and recall of each class
        """
        if not self.detections_done:
            self.run_detection()
        res = []
        for frame in self.frames:
            tp, fp, fn = frame.score(class_id=class_id, threshold=self.bbox_thresh)
            res.append([tp, fp, fn])
        res = np.array(res)
        res_sum = res.sum(axis=0)
        precision = res_sum[0] / (res_sum[0] + res_sum[1])
        recall = res_sum[0] / (res_sum[0] + res_sum[2])
        print('---------------------')
        print(f'Precision for {labelname} = {precision}')
        print(f'Recall for {labelname} = {recall}')
        print('----------------------')
        return precision, recall

    def evaluate_on_dataset(self):
        map_list = []
        for labelname, class_id in self.class_map.items():
            pre, re = self.evaluate_class(class_id=class_id, labelname=labelname)
            map_list.append(pre)

        print(f"MAP:{np.mean(map_list)}")

    def write_annotated_images(self, output_directory='output_frames',annotate_true_bb=True,annotate_pred_bb=True):
        output_directory = os.path.abspath(output_directory)
        if (not os.path.exists(output_directory)):
            os.makedirs(output_directory)

        if not self.detections_done:
            self.run_detection()
        for frame in self.frames:
            annotated_image = frame.annotate(annotate_true_bb,annotate_pred_bb)
            frame_root, frame_fname = os.path.split(frame.frame_name)
            frame_fname_new = os.path.splitext(frame_fname)[0] + '_annotated' + os.path.splitext(frame_fname)[1]
            annotated_fname = os.path.join(output_directory, frame_fname_new)
            cv2.imwrite(annotated_fname, annotated_image)


class ResnetDetector:

    def __init__(self, model_path, score_thresh=0.5, scale=1.0):
        self.model_path = model_path
        self.score_thresh = score_thresh
        self.scale = scale
        self.load_model()

    def load_model(self):
        """
        Loads the model to memory
        """
        self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)

    def detect(self, img):
        """
        Predict detected bboxes on the image using the model, thresholded by score_thresh attribute.
        """
        if not self.model:
            self.load_model()
        scale = self.scale
        bb_list = []
        img_scaled = cv2.resize(img, None, fx=scale, fy=scale)
        img_scaled = preprocess_image(img_scaled)
        _, _, detections = self.model.predict_on_batch(np.expand_dims(img_scaled, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
        detections[:, :4] /= scale
        for idx, (label_idx, score) in enumerate(zip(predicted_labels, scores)):
            if score < self.score_thresh:
                continue
            b = detections[0, idx, :4].astype(int)
            bb_list.append([b[0], b[1], b[2], b[3], label_idx])
        return np.array(bb_list)
