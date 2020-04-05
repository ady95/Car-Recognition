import os
import time

import cv2
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar

from keras.preprocessing.image import ImageDataGenerator

# from utils import load_model
# from resnet_152 import resnet152_model
from resnet_50 import resnet50_model

BASE_FOLDER_PATH = r"D:\DATA\@car\car_fake\test"
IMG_WIDTH, IMG_HEIGHT = 224, 224

def load_model():
    model_weights_path = 'models/model.09-0.98.hdf5'
    # img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 3
    # num_classes = 196
    # model = resnet152_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    model = resnet50_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]



if __name__ == '__main__':
    
    model = load_model()
    num_samples = 14
    img_width, img_height = 224, 224
    # batch_size = 4
    # valid_data = r"C:\Users\beyon\Desktop\valid"
    # valid_data_gen = ImageDataGenerator()
    # valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size, classes=['real','media','monitor'],
    #                                                      shuffle= False, class_mode='categorical', save_to_dir=r'C:\Users\beyon\Desktop\test')
    # valid_generator.reset()
    # print("--Predict--")
    # output = model.predict_generator(valid_generator, steps=31, verbose=1)
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    # print(valid_generator.class_indices)
    # # print(output)
    # # class_label_map = valid_generator.class_indices
    # result=[0,0,0]
    # for res in output:
    #     idx = np.argmax(res)
    #     result[idx] += 1
    # print(result)


    pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    out = open('result.txt', 'a')

    for i in range(num_samples):
        i = 0
        for (path, dir, files) in os.walk(BASE_FOLDER_PATH):
            for filename in files:
                file_path = os.path.join(path, filename)
                if "output" in file_path:
                    continue

                # filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
                bgr_img = cv2.imread(file_path)
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                test_img = np.zeros((4, img_height, img_width, 3), dtype='uint8')
                for i in range(4):
                    crop = random_crop(rgb_img, (img_width, img_height))
                    test_img[i]= crop 
                # rgb_img = cv2.resize(rgb_img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                # rgb_img = np.expand_dims(rgb_img, 0)

                preds = model.predict(test_img)
                print(preds)
                prob = np.max(preds,axis=1)
                print(prob)
                class_id_batch = np.argmax(preds, axis=1)
                bincount = np.bincount(class_id_batch)
                class_id = np.argmax(bincount)
                print(class_id_batch)
                print(bincount)
                
                # out.write(f'{class_id} {file_path} \n')
                # pb.print_progress_bar((i + 1) * 100 / num_samples)

                print(file_path, str(class_id))

                i += 1

    end = time.time()
    seconds = end - start
    print('avg fps: {}'.format(str(num_samples / seconds)))

    out.close()
    K.clear_session()
