import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from baseline import Stream
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class GRU(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru1 = tf.keras.layers.GRU(200, dropout=0.5, return_sequences=False)
        self.fc1 = tf.keras.layers.Dense(200)
        self.fc2 = tf.keras.layers.Dense(4)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
    
    def call(self, x):
        x = self.embed(x)
        x = self.relu(self.bn1(self.gru1(x)))
        x = self.relu(self.bn2(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        return x

if __name__ == '__main__':
    stream = Stream(GRU, 28, 50, 128, 'gru')
    stream.run()