import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB3, EfficientNetB7

import tensorflow_hub as hub

from .metrics import f1_metric
from .utils import get_loss


SEED = 1


def dataaug_layer(rotation, zoom, translation, img_height, img_width, num_channels, seed=SEED):
    
    data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip('horizontal', seed=seed, input_shape=(img_height, img_width, num_channels)),
                                             layers.experimental.preprocessing.RandomRotation(rotation, seed=seed, fill_mode='constant'),
                                             layers.experimental.preprocessing.RandomZoom(zoom, seed=seed, fill_mode='constant'),
                                             layers.experimental.preprocessing.RandomTranslation(translation, translation, seed=seed, fill_mode='constant')])
    
    return data_augmentation


def get_model_simple(loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=1, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    model_simple = Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                               
                               layers.Conv2D(16, 3, activation='relu', input_shape=(img_height, img_width, num_channels)),
                               layers.MaxPooling2D(),
                               layers.Conv2D(32, 3, activation='relu'),
                               layers.MaxPooling2D(),
                               layers.Conv2D(64, 3, activation='relu'),
                               layers.MaxPooling2D(),
                               layers.GlobalMaxPooling2D(),
                               layers.Dense(32, activation='relu'),
                               layers.Dense(num_classes)])
    
    if if_dataaug:
        model_simple = Sequential([data_augmentation,
                                   model_simple])
    
    model_simple.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model_simple


def get_model_tiny(loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=1, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    model_tiny = Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                             
                             layers.Conv2D(64, (5,5), input_shape=(img_height, img_width, num_channels)),
                             layers.BatchNormalization(),
                             layers.Activation(tf.keras.activations.relu),
                             layers.MaxPooling2D((3,3), strides=(2,2)),

                             layers.Conv2D(128, (5,5)),
                             layers.BatchNormalization(),
                             layers.Activation(tf.keras.activations.relu),
                             layers.MaxPooling2D((3,3), strides=(2,2)),

                             layers.Conv2D(256, (5,5)),
                             layers.BatchNormalization(),
                             layers.Activation(tf.keras.activations.relu),
                             layers.MaxPooling2D((3,3), strides=(2,2)),

                             layers.Conv2D(512, (5,5)),
                             layers.BatchNormalization(),
                             layers.Activation(tf.keras.activations.relu),
                             layers.MaxPooling2D((3,3), strides=(2,2)),

                             layers.GlobalAveragePooling2D(),
                             layers.Dense(num_classes)])

    if if_dataaug:
        model_tiny = Sequential([data_augmentation,
                                 model_tiny])

    model_tiny.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model_tiny


def get_model_small(loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=1, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    model_small = Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                             
                              layers.Conv2D(32, (7,7), input_shape=(img_height, img_width, num_channels)),
                              layers.BatchNormalization(),
                              layers.Activation(tf.keras.activations.relu),
                              layers.MaxPooling2D((3,3), strides=(2,2)),

                              layers.Conv2D(64, (7,7)),
                              layers.BatchNormalization(),
                              layers.Activation(tf.keras.activations.relu),
                              layers.MaxPooling2D((3,3), strides=(2,2)),

                              layers.Conv2D(128, (7,7)),
                              layers.BatchNormalization(),
                              layers.Activation(tf.keras.activations.relu),
                              layers.MaxPooling2D((3,3), strides=(2,2)),

                              layers.Conv2D(256, (7,7)),
                              layers.BatchNormalization(),
                              layers.Activation(tf.keras.activations.relu),
                              layers.MaxPooling2D((3,3), strides=(2,2)),

                              layers.GlobalAveragePooling2D(),
                              layers.Dense(num_classes)])

    if if_dataaug:
        model_small = Sequential([data_augmentation,
                                  model_small])

    model_small.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model_small


def get_model_largew(loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=1, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    model_largew = Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                             
                               layers.Conv2D(64, (7,7), input_shape=(img_height, img_width, num_channels)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.Conv2D(128, (7,7)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.Conv2D(256, (7,7)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.Conv2D(512, (7,7)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.GlobalAveragePooling2D(),
                               layers.Dense(num_classes)])

    if if_dataaug:
        model_largew = Sequential([data_augmentation,
                                   model_largew])

    model_largew.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model_largew


def get_model_larget(loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=1, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    model_larget = Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                             
                               layers.Conv2D(64, (5,5), input_shape=(img_height, img_width, num_channels)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.Conv2D(128, (5,5)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.Conv2D(256, (5,5)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.Conv2D(512, (5,5)),
                               layers.BatchNormalization(),
                               layers.Activation(tf.keras.activations.relu),
                               layers.MaxPooling2D((3,3), strides=(2,2)),

                               layers.GlobalAveragePooling2D(),
                               layers.Dense(num_classes)])

    if if_dataaug:
        model_larget = Sequential([data_augmentation,
                                   model_larget])

    model_larget.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model_larget


class MyBiTModel(tf.keras.Model):
    """BiT model with a new head."""

    def __init__(self, num_classes, module):
        super().__init__()

        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(num_classes, kernel_initializer='zeros')
        self.bit_model = module

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


def get_model_bit(model_url='https://tfhub.dev/google/bit/s-r50x1/1', loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], num_classes=3):
    module = hub.KerasLayer(model_url)
    
    model_bit = MyBiTModel(num_classes=num_classes, module=module)

    model_bit.compile(optimizer='adam', loss=loss, metrics=metrics)

    return model_bit


def get_efficientnet_b0(weights='imagenet', loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=3, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    inputs = layers.Input(shape=(img_height, img_width, num_channels))
    
    if weights != 'imagenet':
        outputs = EfficientNetB0(include_top=True, weights=weights, classes=num_classes, classifier_activation=None)(inputs)
        
    else:
        model = EfficientNetB0(include_top=False, input_tensor=inputs, weights=weights)
        model.trainable = False

        x = layers.GlobalAveragePooling2D()(model.output)
#         x = layers.BatchNormalization()(x)

#         top_dropout_rate = 0.2
#         x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
        
    if if_dataaug:
        model = Sequential([data_augmentation,
                            model])        
    
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model


def get_efficientnet_b3(weights='imagenet', loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=3, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    inputs = layers.Input(shape=(img_height, img_width, num_channels))
    
    if weights != 'imagenet':
        outputs = EfficientNetB3(include_top=True, weights=weights, classes=num_classes, classifier_activation=None)(inputs)
        
    else:
        model = EfficientNetB3(include_top=False, input_tensor=inputs, weights=weights)
        model.trainable = False

        x = layers.GlobalAveragePooling2D()(model.output)
#         x = layers.BatchNormalization()(x)

#         top_dropout_rate = 0.2
#         x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
        
    if if_dataaug:
        model = Sequential([data_augmentation,
                            model])        
    
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model


def get_efficientnet_b7(weights='imagenet', loss=get_loss(), metrics=['accuracy', f1_metric(num_classes=3)], if_dataaug=False, num_channels=3, num_classes=3, data_augmentation=None, img_height=512, img_width=512):
    
    inputs = layers.Input(shape=(img_height, img_width, num_channels))
    
    if weights != 'imagenet':
        outputs = EfficientNetB7(include_top=True, weights=weights, classes=num_classes, classifier_activation=None)(inputs)
        
    else:
        model = EfficientNetB7(include_top=False, input_tensor=inputs, weights=weights)
        model.trainable = False

        x = layers.GlobalAveragePooling2D()(model.output)
#         x = layers.BatchNormalization()(x)

#         top_dropout_rate = 0.2
#         x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(num_classes)(x)

    model = tf.keras.Model(inputs, outputs)
        
    if if_dataaug:
        model = Sequential([data_augmentation,
                            model])        
    
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    
    return model