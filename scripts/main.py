import cv2
from deploy_config import endpoint_name
from sagemaker import Predictor
import json


predictor = Predictor(endpoint_name=endpoint_name)
path='test.jpg'
with open(path, 'rb') as f:
    img_bytes = f.read()
payload = bytearray(img_bytes )
byte_data= predictor.predict(payload)
mrz_result=json.loads(byte_data.decode('utf-8'))["mrz"]
assert mrz_result=="P<D<<MUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<\nC01X00T478D<<6408125F2702283<<<<<<<<<<<<<<<4"