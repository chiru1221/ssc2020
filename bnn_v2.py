import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
import os
import random
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

class BNN(tf.keras.Model):
    def __init__(self, sample_num, seed=0):
        super().__init__()
        self.seed = seed
        # self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        
        def kernel_posterior_tensor_fn(d):
            tf.random.set_seed(self.seed)
            return d.sample(sample_num, seed=self.seed)
        
        def bias_posterior_tensor_fn(d):
            tf.random.set_seed(self.seed)
            return d.sample(seed=self.seed)
        
        self.fc1 = tfp.layers.DenseFlipout(400, kernel_posterior_tensor_fn=kernel_posterior_tensor_fn, bias_posterior_tensor_fn=bias_posterior_tensor_fn)
        self.fc2 = tfp.layers.DenseFlipout(400, kernel_posterior_tensor_fn=kernel_posterior_tensor_fn, bias_posterior_tensor_fn=bias_posterior_tensor_fn)
        self.fc3 = tfp.layers.DenseFlipout(4, kernel_posterior_tensor_fn=kernel_posterior_tensor_fn, bias_posterior_tensor_fn=bias_posterior_tensor_fn)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, x):
        # x = self.embed(x)
        # x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        x = tf.reduce_mean(x, axis=0)
        return x

class Stream:
    def __init__(self, model, epoch, batch, name, lr=0.001, cv_num=5, sample_num=0, is_us=True, is_train=True, seed=0, early_stop=None):
        self.model = model
        self.epoch = epoch
        self.batch = batch
        self.name = name
        self.cv_num = cv_num
        self.sample_num = sample_num
        self.is_us = is_us
        self.is_train = is_train
        self.seed = seed
        self.early_stop = early_stop
        
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # self.train_score = tf.keras.metrics.Mean()
        self.train_score = tf.keras.metrics.SparseCategoricalAccuracy()
        self.test_score = tf.keras.metrics.Mean()
    
    def read_data(self):
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sub_df = pd.read_csv('submit_sample.csv', header=None)
        return train_df, test_df, sub_df
    
    def preprocess(self, train_df, test_df):
        stemmer = PorterStemmer()
        
        def apply_preprocess(x):
            x = re.sub('[^a-zA-Z]', ' ', x)
            x = x.split()
            x = [word for word in x if len(word) >= 3]
            x = [stemmer.stem(word) for word in x]
            text = ' '.join(x)
            return text
        
        sentences_train = train_df['description'].apply(apply_preprocess)
        sentences_test = test_df['description'].apply(apply_preprocess)

        sentences = pd.concat([sentences_train, sentences_test])
        bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=2000, stop_words='english')
        # bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
        bow = bow_vectorizer.fit_transform(sentences)
        train = bow[:len(train_df)]
        test = bow[len(train_df):]

        return train, test
    
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
            # self.train_score(neg_log_likelyhood)
            self.train_score(labels, preds)

    def test(self, model, x, y):
        test_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.batch)
        for inputs, labels in test_ds:
            preds = model(inputs)
            acc = f1_score(labels.numpy(), np.argmax(preds.numpy(), axis=1), average='macro')
            self.test_score(acc)
    
    def under_sample(self, x, y):
        label_num = {label: len(np.where(y == label)[0]) for label in np.unique(y)}
        under_num = np.min(list(label_num.values()))
        under_idx = list()
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            np.random.seed(seed=self.seed)
            np.random.shuffle(idx)
            under_idx.extend(idx[:under_num].tolist())
        x = x[under_idx]
        y = y[under_idx]
        return x, y
    
    def cv(self, x, y):
        kf = StratifiedKFold(n_splits=self.cv_num)
        def map_cv(cv_idx, idx):
            model = self.model(self.sample_num, self.seed)
            train_idx, test_idx = idx
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.is_us:
                x_train, y_train = self.under_sample(x_train, y_train)
            
            print('cv:{0:2d}'.format(cv_idx))
            loss = list()
            for epoch in range(self.epoch):
                self.train_score.reset_states()
                self.test_score.reset_states()
                self.train(model, x_train, y_train)
                self.test(model, x_test, y_test)
                print('epoch:{0:3d},train:{1:.4f},test:{2:.4f}'.format(epoch, self.train_score.result().numpy(), self.test_score.result().numpy()))
                loss.append(self.test_score.result().numpy())
                if self.early_stop is not None:
                    if self.early_stop[cv_idx] == epoch:
                        print('train early stop in {0} epoch'.format(epoch))
                        break
            print(np.max(loss), np.argmax(loss))
            model.save_weights('model/' + self.name + '_{0}'.format(cv_idx))
            return self.test_score.result().numpy()
        
        loss = list(map(map_cv, range(self.cv_num), kf.split(x, y)))
        return loss
    
    def predict(self, x, sub_df):
        preds = np.zeros((len(sub_df), 4))
        for cv_idx in range(self.cv_num):
            model = self.model(self.sample_num, self.seed)
            model.load_weights('model/' + self.name + '_{0}'.format(cv_idx))
            preds += model(x).numpy()
        sub_df.iloc[:, 1] = np.argmax(preds, axis=1) + 1
        sub_df.to_csv('submit/' + self.name + '.csv', index=None, header=None)
    
    def run(self):
        train_df, test_df, sub_df = self.read_data()
        train, test = self.preprocess(train_df, test_df)
        
        train = train.toarray()
        test = test.toarray()
        train, test = train.astype(np.float64), test.astype(np.float64)

        if self.is_train:
            loss = self.cv(train, train_df['jobflag'].values - 1)
            print('cv test:{0:.4f}'.format(np.mean(loss)))

        self.predict(test, sub_df)

        
if __name__ == '__main__':
    stream = Stream(BNN, 50, 128, 'bnn_v2', lr=0.0005, sample_num=10, seed=0, early_stop=None)
    stream.run()
    '''
    0 : (50, 128, lr=0.0005, sample_num=10, seed=0, early_stop=None)0.5105
    1 : 
    2 : 
    '''
    
