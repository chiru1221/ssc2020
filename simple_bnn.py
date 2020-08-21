import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import gensim
import os
import random
from baseline import Stream
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


class BNN(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, sample_num, seed=0):
        super().__init__()
        self.seed = seed
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.fc1 = tfp.layers.DenseFlipout(400, kernel_posterior_tensor_fn=lambda d: d.sample(sample_num, seed=self.seed))
        self.fc2 = tfp.layers.DenseFlipout(400, kernel_posterior_tensor_fn=lambda d: d.sample(sample_num, seed=self.seed))
        self.fc3 = tfp.layers.DenseFlipout(4, kernel_posterior_tensor_fn=lambda d: d.sample(sample_num, seed=self.seed))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, x):
        x = self.embed(x)
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        x = tf.reduce_mean(x, axis=0)
        return x

class BNNStream(Stream):
    def __init__(self, model, embed_dim, epoch, batch, name, lr=0.001, cv_num=5, sample_num=0, is_us=True, is_train=True, seed=0):
        super().__init__(model, embed_dim, epoch, batch, name, lr, cv_num, sample_num, is_us, is_train, seed)
    
    def train(self, model, x, y):
        tf.random.set_seed(self.seed)
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.batch).shuffle(100, seed=self.seed)
        for inputs, labels in train_ds:
            with tf.GradientTape() as tape:
                preds = model(inputs)
                
                neg_log_likelyhood = self.loss_fn(labels, preds)
                kl_loss = sum(model.losses) / len(inputs)
                loss = neg_log_likelyhood + kl_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            self.train_score(loss)


if __name__ == '__main__':
    # stream = BNNStream(BNN, 50, 100, 128, 'simple_bnn', lr=0.0005, sample_num=10)
    stream = BNNStream(BNN, 50, 100, 128, 'simple_bnn', lr=0.001, sample_num=10, is_train=True)
    stream.run()
