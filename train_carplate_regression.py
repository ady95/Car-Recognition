import keras
from resnet_152 import resnet152_model
# from resnet_50 import resnet50_model
from resnet_50_regression import resnet50_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import datasets_regression as datasets


img_width, img_height = 224, 224
num_channels = 3
# train_data = r"D:\DATA\@car\car_brand_encar\@back3\train"
# valid_data = r"D:\DATA\@car\car_brand_encar\@back3\valid"
train_folder = r"D:\DATA\@car\car_plate\regression\train"

num_regression_values = 8
# num_train_samples = 1066
# num_valid_samples = 266
verbose = 1
batch_size = 16
num_epochs = 500
patience = 50

if __name__ == '__main__':
    # build a classifier model
    # model = resnet152_model(img_height, img_width, num_channels, num_classes)
    # model = resnet50_model(img_height, img_width, num_channels, num_classes)
    model = resnet50_model(img_height, img_width, num_channels, num_regression_values)


    images, points_array = datasets.load_data(train_folder)
    split = train_test_split(images, points_array, test_size=0.2, random_state=42)
    (trainX, validX, trainY, validY) = split
    
    # images, points_array1, points_array2 = datasets.load_data(train_folder)

    # trainX = images[:num_train_samples]
    # validX = images[num_train_samples:]
    # trainY1 = points_array1[:num_train_samples]
    # validY1 = points_array1[num_train_samples:]
    # trainY2 = points_array2[:num_train_samples]
    # validY2 = points_array2[num_train_samples:]
    
    # print(trainX[0])
    # print(trainY[0])
    # exit()

    # callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs_carplate_regression', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs_carplate_regression/training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    # early_stop = EarlyStopping('val_mean_squared_error', patience=patience)
    # reduce_lr = ReduceLROnPlateau('val_mean_squared_error', factor=0.1, patience=int(patience / 4), verbose=1)
    trained_models_path = 'models_carplate_regression/model'
    # model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    # model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    # callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]
    callbacks = [tensor_board, model_checkpoint, csv_logger]

    print("[INFO] training model...")
    # model.fit(trainX, {"output1":trainY1, "output2":trainY2}, validation_data=(validX, {"output1":validY1, "output2":validY2}), 
    model.fit(trainX, trainY, validation_data=(validX, validY), 
            epochs=num_epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=verbose)

    # model.fit_generator(batch_generator(trainX, trainY, 100, 1),
    #                               steps_per_epoch=300, 
    #                               epochs=10,
    #                               validation_data=batch_generator(validX, validY, 100, 0),
    #                               validation_steps=200,
    #                               verbose=1,
    #                               shuffle = 1)
