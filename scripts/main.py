import requests
import base64
import os
import numpy as np
import mysql.connector
from datetime import datetime,date


class Passport_ExtractLoadDB:


    def __init__(self):
        self.lambda_url = "https://tjbkngoydd.execute-api.us-east-1.amazonaws.com/test"

    @staticmethod
    def calculate_age(birth_date):
        today = date.today()
        age = today.year - birth_date.year - (
                (today.month, today.day) < (birth_date.month, birth_date.day)
        )
        return age

    @staticmethod
    def parse_date(date_str, is_expiry=False):
        current_year_full = datetime.now().year
        current_year = current_year_full % 100

        try:
            year = int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])

            if is_expiry:
                # Expiry dates are future-oriented
                century = 2000 if year <= current_year + 10 else 1900
            else:
                # Date of birth should be in the past
                century = 1900 if year > current_year else 2000

            return date(century + year, month, day)
        except ValueError:
            return None

    @staticmethod
    def parse_mrz(mrz_lines):
        line1, line2 = mrz_lines.split("\n")

        # Extract name
        name_section = line1[5:44]
        names = name_section.split("<<")
        surname = names[0].replace('<', ' ').strip()
        given_names = names[1].replace('<', ' ').strip() if len(names) > 1 else ""

        # Extract MRZ fields
        dob_raw = line2[13:19]
        gender = line2[20]
        expiry_raw = line2[21:27]


        dob = Passport_ExtractLoadDB.parse_date(dob_raw)
        expiry = Passport_ExtractLoadDB.parse_date(expiry_raw, is_expiry=True)
        age = Passport_ExtractLoadDB.calculate_age(dob) if dob else None

        gender_full = {'M': 'Male', 'F': 'Female', '<': 'Unspecified'}.get(gender, 'Unknown')

        return {
            "Surname": surname,
            "Given Names": given_names,
            "Date of Birth": dob,
            "Age": age,
            "Gender": gender_full,
            "Date of Expiry": expiry
        }


    def load_to_db(self,data_dict):
        if len(data_dict.keys())==0:
            return
        photo=np.array(data_dict["photo"])
        data_dict=self.parse_mrz(data_dict["mrz"])
        data_dict["photo"]=photo
        try:

            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password=os.getenv('MYSQL_CONNECTOR_PASSWORD'),
                database="passport_extraction"
            )
            if conn.is_connected():
                print("Connected to MySQL successfully!")

            cursor = conn.cursor()

        except Exception as e:
            print(' Error in mysql connection:', e)

        insert_query = """
            INSERT INTO passport_extraction.passport_extraction_table (`surname`, `given_names`,`date_of_birth`,`age`,`gender`,`date_of_expiry`,`photo`)
        VALUES (%s, %s, %s,%s, %s, %s, %s);
            """

        cursor.execute(insert_query, (data_dict["Surname"],data_dict["Given Names"],data_dict["Date of Birth"]
                                      ,data_dict["Age"],data_dict["Gender"],data_dict["Date of Expiry"],data_dict["photo"].tobytes()))
        print("Latest values injected to DB")
        conn.commit()
        cursor.close()
        conn.close()


    def infer_from_lambda(self,img_path):
        '''

            Args:
                input_data (str): input path for image

            Returns:
                return a dict or empty list

        '''


        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        img_bytes = base64.b64encode(img_bytes).decode('utf-8')
        data = {"data": img_bytes}
        response = requests.post(self.lambda_url, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed with status code {response.status_code}")
            return {}

    def extract(self,img_path):
        return self.infer_from_lambda(img_path)



if __name__=="__main__":

    from glob import glob

    img_paths=sorted(glob("test-samples/*"))
    for img_path in img_paths:
        #img_path="test.jpg"
        print(img_path)
        client=Passport_ExtractLoadDB()
        output=client.infer_from_lambda(img_path)
        client.load_to_db(output)
