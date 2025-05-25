import os
from passport_detector import PassportDetector
from ocr_inference import OCR_Inference
import cv2



class Passport_Extractor():

    def __init__(self,model_dir):
        self.passport_detector_client = PassportDetector(model_path=os.path.join(model_dir,"passport.onnx"))
        self.ocr_client = OCR_Inference(model_path=os.path.join(model_dir,"ocr.xml"))

    @staticmethod
    def split_mrz(mrz_roi):
        h,w,_=mrz_roi.shape
        split_h=h//2
        return [mrz_roi[:split_h,:,:],mrz_roi[split_h:,:,:]]

    def extract(self,img):
        src_img = img.copy()
        detections = self.passport_detector_client.detect(img)
        result={}
        for box, score, label, class_name in zip(detections["boxes"], detections["scores"], detections["classes"],
                                                 detections["classes_names"]):
            box = [int(i) for i in box]
            roi = src_img[box[1]:box[3], box[0]:box[2]]
            if class_name=="photo":
                result[class_name] =roi.tolist()
            if class_name == "mrz":
                texts=[]
                for split_roi in self.split_mrz(roi):
                    texts.append(self.ocr_client.predict([split_roi])[0])
                texts="\n".join(texts)
                result[class_name]=texts
        return result


if __name__=="__main__":
    import json
    import numpy as np
    import base64
    img=cv2.imread("test.jpg")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    passport_extractor_client=Passport_Extractor(os.path.join(project_root,"models"))
    output=passport_extractor_client.extract(img)
    #byte_data = bytearray(json.dumps(output).encode("utf-8"))
    print(passport_extractor_client.extract(img)["mrz"])
    #decoded_array = np.frombuffer(base64.b64decode(byte_data))
    #print(json.loads(byte_data.decode('utf-8'))["mrz"])
    # cv2.imshow("output",passport_extractor_client.passport_detector_client.origin_img)
    # cv2.waitKey(0)