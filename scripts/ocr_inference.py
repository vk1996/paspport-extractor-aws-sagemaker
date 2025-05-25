'''
Copyright 2025 Vignesh(VK)Kotteeswaran <iamvk888@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


import numpy as np
from openvino.runtime import Core
import math
import cv2



def img_decode(img):

    '''

      Converts byte array to numpy array

        Args:
            img(byte array)

        Returns:
            img (numpy array)



    '''
    img = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(img, 1)
    # print(img.shape)

    return img


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                                                                 batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            # print('\n char_list:',char_list)
            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):

        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

        print('\n decoder:', character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character




class OCR_Inference():

    def __init__(self,model_path):

        '''
            Args:
                mode_path(string): path of openvino xml of model
        '''

        ie = Core()

        print('\n',model_path)

        model = ie.read_model(model=model_path)
        self.compiled_model = ie.compile_model(model=model, device_name="CPU")

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.decoder=CTCLabelDecode("models/dict.txt",True)
        self.show_frame=None
        self.image_shape=None
        self.dynamic_width=False




    def resize_norm_img(self,img):

        '''

            Args:

                img : numpy array


            Returns:
                returns preprocessed & normalized numpy array of image
        '''


        self.image_shape=[3,48,int(img.shape[1]*2)]


        imgC,imgH,imgW=self.image_shape


        max_wh_ratio = imgW * 1.0 / imgH
        h, w = img.shape[0], img.shape[1]
        ratio = w * 1.0 / h
        max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
        imgW = int(imgH * max_wh_ratio)

        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))

 
        self.show_frame=resized_image
        resized_image = resized_image.astype('float32')
        
        

        if self.image_shape[0] == 1:
            resized_image = resized_image / 255
            
            resized_image = resized_image[np.newaxis, :]
        else:
            
            resized_image = resized_image.transpose((2, 0, 1)) / 255

        
        
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def predict(self,src):

        '''

            Args:
                src : either list of images numpy array or list of image filepath string

            Returns:

                list of texts


        '''

        imgs=[]
        show_frames=[]

        for item in src:

            if hasattr(item,'shape'):

                imgs.append(np.expand_dims(self.resize_norm_img(item),axis=0))

            elif isinstance(item,str):
                
                with open(item, 'rb') as f:
                    content=f.read()
                imgs.append(np.expand_dims(self.resize_norm_img(img_decode(content)),axis=0))

            else:
                return "Error: Invalid Input"
            
            show_frames.append(self.show_frame)
            

        blob=np.concatenate(imgs,axis=0).astype(np.float32)
        
        outputs = self.compiled_model([blob])[self.output_layer]

        
        texts=[]

        for output in outputs:

            output=np.expand_dims(output,axis=0)

            curr_text=self.decoder(output)[0][0]


            texts.append(curr_text)

        
   
        return texts
    


    
        



