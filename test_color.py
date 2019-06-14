import os
import time

import cv2
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar

# from utils import load_model
# from resnet_152 import resnet152_model
from resnet_50 import resnet50_model

BASE_FOLDER_PATH = r"D:\DATA\car_corlor\valid"
IMG_WIDTH, IMG_HEIGHT = 224, 224

def load_model():
    model_weights_path = 'models/model.25-0.66.hdf5'
    # img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 13
    # num_classes = 196
    # model = resnet152_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    model = resnet50_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model





if __name__ == '__main__':
    
    model = load_model()
    num_samples = 211

    pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    out = open('result.txt', 'a')

    # for i in range(num_samples):

    i = 0
    for (path, dir, files) in os.walk(BASE_FOLDER_PATH):
        for filename in files:
            file_path = os.path.join(path, filename)
            if "output" in file_path:
                continue

            # filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
            bgr_img = cv2.imread(file_path)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            rgb_img = np.expand_dims(rgb_img, 0)

            preds = model.predict(rgb_img)
            prob = np.max(preds)
            class_id = np.argmax(preds)
            out.write(f'{class_id} {file_path} \n')
            pb.print_progress_bar((i + 1) * 100 / num_samples)

            print(file_path, str(class_id))

            i += 1

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()
    K.clear_session()
