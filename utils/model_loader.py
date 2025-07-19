import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications import efficientnet, vgg16, resnet50, resnet

@tf.keras.utils.register_keras_serializable()
class PreProcessingLayer(Layer):
    def __init__(self, model_name, **kwargs):
        super(PreProcessingLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.preprocess = {
            "EfficientNetB0": efficientnet.preprocess_input,
            "VGG16": vgg16.preprocess_input,
            "ResNet50": resnet50.preprocess_input,
            "ResNet101": resnet.preprocess_input
        }

    def call(self, inputs):
        return self.preprocess.get(self.model_name)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

@st.cache_resource
def load_trained_model(model_name):
    model_path = f"models/best_{model_name}.keras"
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"PreProcessingLayer": PreProcessingLayer}
    )

# Preprocessing helper
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32)  # No /255.0 normalization
    return np.expand_dims(arr, axis=0)