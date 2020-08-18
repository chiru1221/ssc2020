import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from simple_bnn import BNNStream
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class BNN(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.conv1 = tfp.layers.Convolution1DFlipout(64, 4)
        self.conv2 = tfp.layers.Convolution1DFlipout(64, 4)
        self.fc1 = tfp.layers.DenseFlipout(4)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, x):
        x = self.embed(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.softmax(self.fc1(x))
        return x

if __name__ == '__main__':
    stream = BNNStream(BNN, 28, 100, 128, 'bnn_cnn', lr=0.0005, sample_num=10)
    stream.run()