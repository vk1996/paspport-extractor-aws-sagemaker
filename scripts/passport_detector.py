import cv2
import numpy as np
import onnxruntime
from collections import defaultdict


class PassportDetector():

    def __init__(self, model_path, scaling=False, classes_path="models/class_names.txt"):
        self.scaling = scaling
        classes_path=model_path.replace("passport.onnx","class_names.txt")
        with open(classes_path, 'r') as classes_file:
                self.classes = classes_file.read().split('\n')
        providers = ['CPUExecutionProvider']
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.image_size = self.model.get_inputs()[0].shape[-2:]
        print("Passport detector model loaded")

    def __preprocess_image(self, img, swap=(2, 0, 1)):
        padded_img = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 114
        r = min(self.image_size[0] / img.shape[0], self.image_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def __scale_coordinates(self, boxes, classes, img_size):
        for i, class_idx in enumerate(classes):
            width_scale_factor = self.classes_layout[class_idx]['w_scale']
            coord_width = boxes[i][2] - boxes[i][0]
            width_gain = ((coord_width * width_scale_factor) - coord_width) / 2
            boxes[i][0] = max(boxes[i][0] - width_gain, 0)
            boxes[i][2] = min(boxes[i][2] + width_gain, img_size[1]-1)
            height_scale_factor = self.classes_layout[class_idx]['h_scale']
            coord_height = boxes[i][3] - boxes[i][1]
            height_gain = ((coord_height * height_scale_factor) - coord_height) / 2
            boxes[i][1] = max(boxes[i][1] - height_gain, 0)
            boxes[i][3] = min(boxes[i][3] + height_gain, img_size[0]-1)
        return boxes

    @staticmethod
    def __new_nms(boxes, scores, iou_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep
    
    @staticmethod
    def __remove_duplicates(coordinates, scores, classes):
        duplicates = defaultdict(list)
        for i, item in enumerate(classes):
            duplicates[item].append(i)
    
        duplicate_indexes = {k: v for k, v in duplicates.items() if len(v) > 1}
        if len(duplicate_indexes) > 0:
            indexes_to_remove = []
            for value in duplicate_indexes.values():
                highest_score_index = -1
                score = -1
                for index in value:
                    if scores[index] > score:
                        score = scores[index]
                        highest_score_index = index
                indexes_to_remove.append([index for index in value if index != highest_score_index])
            coordinates = np.delete(coordinates, indexes_to_remove, 0)
            scores = np.delete(scores, indexes_to_remove)
            classes = np.delete(classes, indexes_to_remove)
        return coordinates, scores, classes

    def __parse_output_data(self, outputs):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32]
        hsizes = [self.image_size[0] // stride for stride in strides]
        wsizes = [self.image_size[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs[0]

    def __decode_prediction(self, prediction, img_size, resize_ratio, score_thresh, iou_thresh):

        boxes = prediction[:, :4]
        classes = prediction[:, 4:5] * prediction[:, 5:]
        scores = np.amax(classes, axis=1)
        classes = np.argmax(classes, axis=1)

        valid_score_mask = scores > score_thresh
        if valid_score_mask.sum() == 0:
            return np.array([]), np.array([]), np.array([])
        valid_scores = scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_classes = classes[valid_score_mask]

        valid_boxes_xyxy = np.ones_like(valid_boxes)
        valid_boxes_xyxy[:, 0] = valid_boxes[:, 0] - valid_boxes[:, 2]/2.
        valid_boxes_xyxy[:, 1] = valid_boxes[:, 1] - valid_boxes[:, 3]/2.
        valid_boxes_xyxy[:, 2] = valid_boxes[:, 0] + valid_boxes[:, 2]/2.
        valid_boxes_xyxy[:, 3] = valid_boxes[:, 1] + valid_boxes[:, 3]/2.
        valid_boxes_xyxy /= resize_ratio

        indices = self.__new_nms(valid_boxes_xyxy, valid_scores, iou_thresh)
        valid_boxes_xyxy = valid_boxes_xyxy[indices, :]
        valid_scores = valid_scores[indices]
        valid_classes = valid_classes[indices].astype('int')

        
        valid_boxes_xyxy, valid_scores, valid_classes = self.__remove_duplicates(valid_boxes_xyxy, valid_scores, valid_classes)

        valid_classes_names=[self.classes[i] for i in valid_classes]
        
        if self.scaling:
            valid_boxes_xyxy = self.__scale_coordinates(valid_boxes_xyxy, valid_classes, img_size)

        return valid_boxes_xyxy, valid_scores, valid_classes,valid_classes_names

    def draw_boxes(self, img, boxes, scores=None, classes=None, class_names=None):

        for i in range(boxes.shape[0]):
            cv2.rectangle(img,
                          (int(boxes[i, 0]), int(boxes[i, 1])),
                          (int(boxes[i, 2]), int(boxes[i, 3])),
                          (0, 128, 0),
                          int(0.005 * img.shape[1]))


            text_label = class_names[i]
            #text_label+= ' ' + str("%.2f" % round(scores[i],2))

            w, h = cv2.getTextSize(text_label, 0, fontScale=0.5, thickness=1)[0]
            cv2.putText(img,
                        text_label,
                        (int(boxes[i,0]) if int(boxes[i,0])+w<img.shape[1] else img.shape[1]-w, int(boxes[i,1])-2 if (int(boxes[i,1])-h>=3) else int(boxes[i,1])+h+2),
                        0,
                        0.5,
                        (0,0,255),
                        thickness= int(0.0025*img.shape[1]),
                        lineType=cv2.LINE_AA)

    def detect(self, image, score_thresh=0.5, iou_thresh=0.5):
        h,w = image.shape[:2]
        self.origin_img = np.copy(image)
        model_input = np.copy(image)
        model_input, resize_ratio = self.__preprocess_image(model_input)
        prediction = self.model.run(None, {self.model.get_inputs()[0].name: model_input[None, :, :, :]})
        prediction = self.__parse_output_data(prediction[0])
        d_boxes, d_scores, d_classes, d_classes_names =self.__decode_prediction(prediction, (h,w), resize_ratio, score_thresh, iou_thresh)
        self.draw_boxes(self.origin_img, d_boxes,None, d_classes,d_classes_names)
        return {"boxes":d_boxes,"scores":d_scores,"classes":d_classes,"classes_names":d_classes_names}
