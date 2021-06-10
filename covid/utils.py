import json
import gc
import shutil
import os
import pathlib

import cv2
import numpy as np
from sklearn.utils import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB3, EfficientNetB7
from tensorflow.keras.models import Sequential

from livelossplot.inputs.tf_keras import PlotLossesCallback


CLASS_NAMES = ['COV', 'Normal', 'OtherPneumonia']
SEED = 1


def get_datasets(path_train, path_val, img_height, img_width, num_channels, batch_size, class_names=CLASS_NAMES, seed=SEED):
    
    if num_channels == 1:
        color_mode = 'grayscale'
    elif num_channels == 3:
        color_mode='rgb'
    
    train_ds = image_dataset_from_directory(path_train,
                                            class_names=class_names,
                                            color_mode=color_mode,
                                            image_size=(img_height, img_width),
                                            seed=seed,
                                            batch_size=batch_size,
                                            label_mode='categorical')

    val_ds = image_dataset_from_directory(path_val,
                                          class_names=class_names,
                                          color_mode=color_mode,
                                          image_size=(img_height, img_width),
                                          seed=seed,
                                          batch_size=batch_size,
                                          label_mode='categorical')


    print(train_ds.class_names)
    print(val_ds.class_names)
    
    return train_ds, val_ds


def get_datagenerators(PATH_TRAIN, PATH_VAL, num_channels, img_width, img_height, batch_size, class_names=CLASS_NAMES, seed=SEED, if_dataaug=False, rotation=0, zoom=0, translation=0):   
    
    if if_dataaug:
        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                        horizontal_flip=True,
                                                                        rotation_range=rotation,
                                                                        width_shift_range=translation,
                                                                        height_shift_range=translation,
                                                                        fill_mode='constant',
                                                                        zoom_range=zoom)

    else:
        datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        
    datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    
    if num_channels == 1:
        color_mode = 'grayscale'
    elif num_channels == 3:
        color_mode='rgb'
    
    
    train_generator = datagen_train.flow_from_directory(PATH_TRAIN,
                                                        classes=class_names,
                                                        color_mode=color_mode,
                                                        target_size=(img_height, img_width),
                                                        seed=seed,
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    
    val_generator = datagen_val.flow_from_directory(PATH_VAL,
                                                    classes=class_names,
                                                    color_mode=color_mode,
                                                    target_size=(img_height, img_width),
                                                    seed=seed,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
    
    
    return train_generator, val_generator


def get_model_checkpoint(filepath, monitor='val_f1_score', save_weights_only=False, save_best_only=True, verbose=1, mode='max'):
    model_checkpoint = ModelCheckpoint(filepath=filepath,
                                       save_weights_only=save_weights_only,
                                       monitor=monitor,
                                       save_best_only=save_best_only, 
                                       verbose=verbose,
                                       mode=mode)
    
    return model_checkpoint


def get_early_stopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1, restore_best_weights=True):
    early_stopping = EarlyStopping(monitor=monitor,
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   restore_best_weights=restore_best_weights)
    
    return early_stopping


def get_loss(from_logits=True):
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    
    return loss


def train_model(get_model, train_ds, val_ds, dir_name, model_name, root, epochs, num_iter, class_weight=None):
    all_history = []
    
    if class_weight:
        if type(train_ds) == tf.keras.preprocessing.image.DirectoryIterator:
            ds = get_datasets('CT-COV19/Train', 'CT-COV19/Val', img_height=512, img_width=512, num_channels=1, batch_size=32)[0]    
            train_labels = np.concatenate([y for x, y in ds], axis=0)
        else:
            train_labels = np.concatenate([y for x, y in train_ds], axis=0)
        
        class_weight = compute_class_weight('balanced', classes=[0, 1, 2], y=np.argmax(train_labels, axis=1)) 
        class_weight = dict(enumerate(class_weight))
        
    if not os.path.isdir('best-models/' + dir_name + '-weights/'):
        os.mkdir('best-models/' + dir_name + '-weights/')
    
    for _ in range(num_iter):
        model = get_model
        model_checkpoint = get_model_checkpoint(root + 'covid-ct/best-models/' + dir_name + '-weights/' + model_name + '.ckpt', save_weights_only=True)

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=[PlotLossesCallback(), model_checkpoint])
        all_history.append(history.history)
    
        if _ + 1 < num_iter:
            del model, history
            gc.collect()

        else:
            if not os.path.isdir('history/' + dir_name + '/'):
                os.mkdir('history/' + dir_name + '/')
            with open('history/' + dir_name + '/' + model_name + '.json', 'w') as out:
                json.dump(all_history, out)


def files_paths_labels(path, subset):
    files_paths = []
    files_labels = []

    for root, dirs, files in os.walk(path):
        p = pathlib.Path(root)

        for file in files:
            if file.split('.')[-1] != 'npy':
                files_paths.append(root + '/' + file)

                if p.parts[-2] == subset: files_labels.append(p.parts[-1])
                else: files_labels.append(p.parts[-2])
    
    print(subset, len(files_paths), len(files_labels))
    
    files_labels = [x for _,x in sorted(zip(files_paths,files_labels), key=lambda pair: pair[0])]
    files_paths = sorted(files_paths)
    
    return files_paths, files_labels


def prepare_X_y(files_paths, files_labels, img_width=512, img_height=512, num_channels=1):
    X = []

    for path in files_paths:
        if num_channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif num_channels == 3:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        if (img.shape[0] or img.shape[1]) != 512:
            img = cv2.resize(img, (img_width, img_height))

        X.append(img)

    X = np.asarray(X)
    y = np.asarray(files_labels)

    return X, y


def preprocess_matrices(X, y):
    if X.ndim == 3 :
        X = np.expand_dims(X, axis=-1)

    y = list(map(lambda x: CLASS_NAMES.index(x), y))
    y = np.asarray(y)

    if y.ndim == 1:
        y = to_categorical(y, len(CLASS_NAMES))

    print('Maximum pixel values: ', np.max(X))
    print('X shape: ', X.shape)
    print('y shape: ', y.shape)
    
    return X, y


def load_efficientnet(img_height, img_width, num_channels=3, num_classes=3, if_dataaug=False, data_augmentation=None, b=3):
    #additional EfficientNet loading function - created due to model loading troubles

    if b==3:
        base_model = EfficientNetB3(weights=None, input_shape=(img_height, img_width, num_channels), include_top=True, classes=3)
        
    if b==0:
        base_model = EfficientNetB0(weights=None, input_shape=(img_height, img_width, num_channels), include_top=True, classes=3)
        
    if b==7:
        base_model = EfficientNetB7(weights=None, input_shape=(img_height, img_width, num_channels), include_top=True, classes=3)

    inputs = layers.Input(shape=(img_height, img_width, num_channels))
    
    model = Sequential([inputs,
                        base_model])
    
    if if_dataaug:
        model = Sequential([data_augmentation,
                            model])
    
    return model