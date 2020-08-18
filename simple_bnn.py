import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gru import GRUStream
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class BNN(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.fc1 = tfp.layers.DenseFlipout(200)
        self.fc2 = tfp.layers.DenseFlipout(200)
        self.fc3 = tfp.layers.DenseFlipout(4)
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
        return x

class BNNStream(GRUStream):
    def __init__(self, model, embed_dim, epoch, batch, name, lr=0.001, cv_num=5, sample_num=0, is_train=True):
        super().__init__(model, embed_dim, epoch, batch, name, lr, cv_num, sample_num, is_train)
    
    def preprocess(self, train_df, test_df):
        word_train = [word for words in train_df['description'].str.split(' ') for word in words]
        word_test = [word for words in test_df['description'].str.split(' ') for word in words]
        word = np.union1d(word_train, word_test)
        # vocab_size = len(word)

        word_to_idx = dict()
        idx = 2
        for w in word:
            if w not in stopwords.words('english'):
                word_to_idx[w] = idx
                idx += 1
            else:
                word_to_idx[w] = 1

        vocab_size = np.max(list(word_to_idx.values()))
        max_word_len = np.max([train_df['description'].str.split(' ').apply(len).max(), test_df['description'].str.split(' ').apply(len).max()])
        train = np.zeros((len(train_df), max_word_len))
        test = np.zeros((len(test_df), max_word_len))

        for i, words in enumerate(train_df['description'].str.split(' ')):
            train[i, :len(words)] = np.array([word_to_idx[word] for word in words])
        for i, words in enumerate(test_df['description'].str.split(' ')):
            test[i, :len(words)] = np.array([word_to_idx[word] for word in words])
        
        return train, test, vocab_size
    
    def train(self, model, x, y):
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.batch).shuffle(100)
        for inputs, labels in train_ds:
            with tf.GradientTape() as tape:
                preds = model(inputs)
                if self.sample_num != 0:
                    for sample in range(self.sample_num - 1):
                        preds += model(inputs)
                    preds /= self.sample_num
                
                neg_log_likelyhood = self.loss_fn(labels, preds)
                kl_loss = sum(model.losses) / len(inputs)
                loss = neg_log_likelyhood + kl_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            self.train_score(loss)


if __name__ == '__main__':
    # stream = BNNStream(BNN, 28, 50, 128, 'simple_bnn', lr=0.0005, sample_num=10)
    stream = BNNStream(BNN, 28, 100, 128, 'simple_bnn_preprocess_v1', lr=0.0005, sample_num=10)
    stream.run()