#Adapted from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


class GradCAM:

    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        self.nested = False
        
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
    #try different approaches depending on the model structure (sequential/functional/nested)

        try:
            if 'random' in self.model.layers[0].layers[0].name:
                for layer in reversed(self.model.layers[1].layers[0].layers):
                    if len(layer.output_shape) == 4:
                        return layer.name
        
        except:
            pass
        
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name    

        for layer in reversed(self.model.layers[0].layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
    #gradModel is constructed depending on the model structure (sequential/functional/nested)

        try:
            gradModel = Model(inputs=self.model.inputs,
                              outputs=[self.model.get_layer(self.layerName).output, self.model.output])
    
        except ValueError:
            try:
                gradModel = Model(inputs=self.model.layers[0].input,
                                  outputs=[self.model.layers[0].get_layer(self.layerName).output, self.model.layers[0].layers[-1].output])
                
            except ValueError:
                gradModel = Model(inputs=self.model.layers[1].layers[0].input,
                                  outputs=[self.model.layers[1].layers[0].get_layer(self.layerName).output, self.model.layers[1].layers[0].layers[-1].output])

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

        return (heatmap, output)
    
    
def show_gradcam(img_path, model, img_params, class_idx=None):
    orig = cv2.imread(img_path)

    if img_params['num_channels'] == 1:
        color_mode = 'grayscale'
    elif img_params['num_channels'] == 3:
        color_mode = 'rgb'

    image = load_img(img_path, target_size=(img_params['img_height'], img_params['img_width']), color_mode=color_mode)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)
    
    if class_idx is None:
        class_idx = np.argmax(preds[0])
        print('Class: ', class_idx)
    
    cam = GradCAM(model, class_idx)
    heatmap = cam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
    
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(30,15))
    plt.subplot(131)
    plt.imshow(orig)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(heatmap)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(output)
    plt.axis('off')
    plt.tight_layout();
