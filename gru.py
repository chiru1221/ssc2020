import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from baseline import Stream
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class GRU(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru1 = tf.keras.layers.GRU(200, dropout=0.5, return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(200, dropout=0.5, return_sequences=False)
        self.fc1 = tf.keras.layers.Dense(4)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
    
    def call(self, x):
        x = self.embed(x)
        x = self.relu(self.bn1(self.gru1(x)))
        x = self.relu(self.bn2(self.gru2(x)))
        x = self.softmax(self.fc1(x))
        return x

class GRUStream(Stream):
    def __init__(self, model, embed_dim, epoch, batch, name, lr=0.001, cv_num=5, sample_num=0, is_train=True):
        super().__init__(model, embed_dim, epoch, batch, name, lr, cv_num, sample_num, is_train)

    def under_sample(self, x, y):
        label_num = {label: len(np.where(y == label)[0]) for label in np.unique(y)}
        under_num = np.min(list(label_num.values()))
        under_idx = list()
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            np.random.shuffle(idx)
            under_idx.extend(idx[:under_num].tolist())
        x = x[under_idx]
        y = y[under_idx]
        return x, y

    def cv(self, x, y, vocab_size):
        kf = StratifiedKFold(n_splits=self.cv_num)
        def map_cv(cv_idx, idx):
            model = self.model(vocab_size, self.embed_dim)
            train_idx, test_idx = idx
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            x_train, y_train = self.under_sample(x_train, y_train)
            
            print('cv:{0:2d}'.format(cv_idx))
            for epoch in range(self.epoch):
                self.train_score.reset_states()
                self.test_score.reset_states()
                self.train(model, x_train, y_train)
                self.test(model, x_test, y_test)
                print('epoch:{0:3d},train:{1:.4f},test:{2:.4f}'.format(epoch, self.train_score.result().numpy(), self.test_score.result().numpy()))

            model.save('model/' + self.name + '_{0}'.format(cv_idx))
            return self.test_score.result().numpy()
        
        loss = list(map(map_cv, range(self.cv_num), kf.split(x, y)))
        return loss

if __name__ == '__main__':
    stream = GRUStream(GRU, 28, 50, 128, 'gru')
    stream.run()