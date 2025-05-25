import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from passport_extractor import  Passport_Extractor
import cv2

def test():
    img=cv2.imread("test.jpg")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    passport_extractor_client = Passport_Extractor(os.path.join(project_root, "models"))
    output=passport_extractor_client.extract(img)
    assert output["mrz"]=="P<D<<MUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<\nC01X00T478D<<6408125F2702283<<<<<<<<<<<<<<<4"