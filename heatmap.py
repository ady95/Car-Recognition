import os
import time

import cv2
import numpy as np

from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras import backend as K

# from utils import load_model
from resnet_152 import resnet152_model
# from resnet_50 import resnet50_model

import common_util


CLASSID_JSON = r"D:\DATA\@car\car_classification\classid_to_folder.json"
# BASE_FOLDER_PATH = r"D:\DATA\@car\car_classification\valid"
# BASE_FOLDER_PATH = r"D:\DATA\@car\car_classification_google\@TEST"
# BASE_FOLDER_PATH = r"D:\DATA\@car\car_classification\test_hyundai\20190528"
BASE_FOLDER_PATH = r"D:\DATA\@car\car_photo\carphoto_20190618"
IMG_WIDTH, IMG_HEIGHT = 224, 224

def load_model():
    model_weights_path = 'models_brand1/model.63-0.93.hdf5'
    # img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 60
    # num_classes = 196
    model = resnet152_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    # model = resnet50_model(IMG_WIDTH, IMG_HEIGHT, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model


# last_conv_layer_name = 'res5c_relu' # 최종단
# last_conv_layer_name = 'res4b35_relu' # shape이 줄기 전의 단

def prepare_single_input(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis= 0) # (1, 224, 224, 3)
    # x = preprocess_input(x) # Imagenet의 데이터는 전처리가 조금 다름. (vgg16에 쓰임)
    return x

def generate_heatmap(model, class_idx, last_conv_layer_name):
    output = model.output[:,class_idx]
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2)) # 이건 바꿔야 할 수 도 있음.
    iterate = K.function([model.input],
                        [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    ch_num = last_conv_layer.output_shape[-1]
    for i in range(ch_num):
        conv_layer_output_value[:,:,i]*=pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    del output
    del last_conv_layer
    del grads
    del pooled_grads
    del pooled_grads_value
    del conv_layer_output_value
    return heatmap

def merge_with_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, 0.5, original_img, 0.5, 0)



if __name__ == '__main__':
    
    CLASSID_DICT = common_util.load_json(CLASSID_JSON)

    model = load_model()


    img_path = r'D:\DATA\@car\car_classification\train\0001 2010 SONATA\000053.jpg'

    # 어느시점에서 영향도를 보기 원하는가?
    last_conv_layer_name = 'res5c_relu' # 최종단
    # last_conv_layer_name = 'res4b35_relu' # shape이 줄기 전의 단


    x = prepare_single_input(img_path, target_size = (224, 224))
    preds = model.predict(x)
    class_index = np.unravel_index(preds.argmax(), preds.shape)[1]
    print(class_index)

    heatmap = generate_heatmap(model, class_index, last_conv_layer_name)
    img = plt.imread(img_path)
    original_img = img[..., ::-1] 
    heatmap_img = merge_with_heatmap(original_img, heatmap)

    plt.figure()
    print(CLASSID_DICT[str(class_index)])
    plt.imshow(heatmap_img)
    plt.show()
    exit()
    # output = model.output[:,class_index]

    # last_conv_layer = model.get_layer(last_conv_layer_name)
    # grads = K.gradients(output, last_conv_layer.output)[0]
    # pooled_grads = K.mean(grads, axis=(0,1,2)) # 이건 바꿔야 할 수 도 있음.
    # iterate = K.function([model.input],
    #                     [pooled_grads, last_conv_layer.output[0]])
    
    # pooled_grads_value, conv_layer_output_value = iterate([x])
    # ch_num = last_conv_layer.output_shape[-1]
    # print(ch_num)
    # print(type(conv_layer_output_value))

    # for i in range(ch_num):
    #     # print(conv_layer_output_value)
    #     conv_layer_output_value[:,:,i]*=pooled_grads_value[i]

    # heatmap = np.mean(conv_layer_output_value, axis=-1)
    # heatmap = np.maximum(heatmap,0)
    # heatmap /= np.max(heatmap)

    # 반복되면 메모리 오류가 발생하므로 모델 제외하고 추가적으로 발생되는 네트워크들을 제거함
    # del output
    # del last_conv_layer
    # del grads
    # del pooled_grads
    # del pooled_grads_value
    # del conv_layer_output_value


    # 시각화
    img = plt.imread(img_path)
    # img = cv2.imread(img_path) #issue; allways returns None
    img = img[..., ::-1] 
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
    plt.figure()
    print(CLASSID_DICT[str(class_index)])
    plt.imshow(superimposed_img)
    plt.show()