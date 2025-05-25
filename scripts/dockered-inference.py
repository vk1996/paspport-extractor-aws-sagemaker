'''

Inference script that runs in AWS Sagemaker endpoint
Receives image / text payload from local
Preprocess image & tokenize text for predicting image & text vectors
'''
import io
import os
import json
from PIL import Image
from passport_extractor import  Passport_Extractor
import logging
from glob import glob
import numpy as np








class Inference(object):

    def __init__(self):

        self.initialized = False
        self.logger = logging.getLogger("ConsoleLogger")
        self.logger.setLevel(logging.INFO)
        self.model=None
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)



    def initialize(self, context):

        self.initialized = True
        properties = context.system_properties
        model_dir= properties.get("model_dir")
        print("[INFO]:",model_dir)
        print(glob(model_dir))
        self.model=Passport_Extractor(model_dir)
        self.logger.info('ONNX Model loaded')

    def check_bytearray_type(self,data: bytearray):


        try:
            image = Image.open(io.BytesIO(data))
            image.verify()  # Verify the image file
            self.logger.info("payload is an image.")
            return "image"
        except Exception as e:
            self.logger.error('Unsupported payload input')
        return "unknown"


    def inference(self,request):

        payload=request[0]['body']
        self.logger.info('Checking payload type')
        type_of_payload=self.check_bytearray_type(payload)

        if type_of_payload=="image":
            self.logger.info('Predicting in image mode')
            data = Image.open(io.BytesIO(payload))
            data = np.array(data)
            self.logger.info('Input shape: '+str(data.shape))
            output=self.model.extract(data)

        else:
            self.logger.error('Incomplete Prediction')
            return []
        self.logger.info('Prediction completed')
        #tensor_bytes = output.tobytes()
        byte_data = bytearray(json.dumps(output).encode("utf-8"))
        self.logger.info('Output ready')
        return [byte_data]



    def handle(self, data, context):

        return self.inference(data)


_service = Inference()



def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)


