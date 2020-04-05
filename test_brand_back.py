import os
import time

import cv2
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar

from keras.preprocessing.image import ImageDataGenerator

# from utils import load_model
from resnet_152 import resnet152_model
# from resnet_50 import resnet50_model

import common_util
import opencv_util

CLASSID_JSON = r"D:\DATA\@car\car_brand_back\classid_to_folder.json"

IMG_WIDTH, IMG_HEIGHT = 224, 224

def load_model():
    # model_weights_path = 'models_brand_encar/model.58-0.95.hdf5'
    model_weights_path = 'models_brand_back/model.57-0.99.hdf5'
    # img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 106

    model = resnet152_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    # model = resnet50_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model





if __name__ == '__main__':
    
    classid_dict = common_util.load_json(CLASSID_JSON)

    model = load_model()


   #########################
    # 일반 테스트
    # BASE_FOLDER_PATH = r"D:\TEMP\sellcarauction_image"
    # # files = ["AT157613393182826.jpg", "AT157613393182826.jpg", "AT157613393182826.jpg", "AT157613393182826.jpg", "AT157613393182826.jpg"]
    # files = os.listdir(BASE_FOLDER_PATH)
    # for filename in files:
    #     file_path = os.path.join(BASE_FOLDER_PATH, filename)

    #     bgr_img = opencv_util.imread(file_path)
    #     rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    #     rgb_img = cv2.resize(rgb_img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    #     rgb_img = np.expand_dims(rgb_img, 0)

    #     preds = model.predict(rgb_img)
    #     # print(preds)
    #     prob = np.max(preds)
    #     class_id = np.argmax(preds)
    #     class_name = classid_dict[str(class_id)]

    #     result_line = f'{filename}\t{class_name}\t{prob}'
    #     print(result_line)

    # exit()

    ################################
    # 테스트 셋 
    BASE_FOLDER_PATH = r"D:\DATA\@car\car_brand_back\test"
    num_samples = 970

    pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    out = open('encar_back_result.txt', 'a')

    # for i in range(num_samples):

    i = 0
    t = 0
    for folder in os.listdir(BASE_FOLDER_PATH):
        folder_path = os.path.join(BASE_FOLDER_PATH, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if "output" in file_path:
                continue

            bgr_img = opencv_util.imread(file_path)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            rgb_img = cv2.resize(rgb_img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            rgb_img = np.expand_dims(rgb_img, 0)

            preds = model.predict(rgb_img)
            prob = np.max(preds)
            class_id = np.argmax(preds)
            class_name = classid_dict[str(class_id)]

            # 정답 여부 확인
            result = 1 if folder == class_name else 0

            result_line = f'{folder}\\{filename}\t{class_name}\t{prob}\t{result}'

            out.write(result_line + '\n')
            pb.print_progress_bar((i + 1) * 100 / num_samples)

            print(result_line)

            i += 1
            t += result
            print(t / i * 100, "%")

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()
    K.clear_session()
