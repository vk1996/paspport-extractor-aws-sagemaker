import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from main import Passport_ExtractLoadDB

def test():
    img_path = "test.jpg"
    client = Passport_ExtractLoadDB()
    output = client.infer_from_lambda(img_path)
    assert output["mrz"] == "P<D<<MUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<\nC01X00T478D<<6408125F2702283<<<<<<<<<<<<<<<4"