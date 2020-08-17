import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
tf.keras.backend.set_floatx('float64')
gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

class Baseline(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.fc1 = tf.keras.layers.Dense(400)
        self.fc2 = tf.keras.layers.Dense(400)
        self.fc3 = tf.keras.layers.Dense(4)
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

class Stream:
    def __init__(self, model, embed_dim, epoch, batch, name, lr=0.001, cv_num=5, sample_num=0, is_train=True):
        self.model = model
        self.embed_dim = embed_dim
        self.epoch = epoch
        self.batch = batch
        self.name = name
        self.cv_num = cv_num
        self.sample_num = sample_num
        self.is_train = is_train
        
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_score = tf.keras.metrics.Mean()
        self.test_score = tf.keras.metrics.SparseCategoricalAccuracy()
    
    def read_data(self):
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        sub_df = pd.read_csv('submit_sample.csv', header=None)
        return train_df, test_df, sub_df
    
    def preprocess(self, train_df, test_df):
        word_train = [word for words in train_df['description'].str.split(' ') for word in words]
        word_test = [word for words in test_df['description'].str.split(' ') for word in words]
        word = np.union1d(word_train, word_test)
        vocab_size = len(word)

        word_to_idx = {word: idx for idx, word in enumerate(word)}
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
                
                loss = self.loss_fn(labels, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            self.train_score(loss)

    def test(self, model, x, y):
        test_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.batch)
        for inputs, labels in test_ds:
            preds = model(inputs)
            if self.sample_num != 0:
                for sample in range(self.sample_num - 1):
                    preds += model(inputs)
                preds /= self.sample_num
            self.test_score(labels, preds)
    
    def cv(self, x, y, vocab_size):
        kf = StratifiedKFold(n_splits=self.cv_num)
        def map_cv(cv_idx, idx):
            model = self.model(vocab_size, self.embed_dim)
            train_idx, test_idx = idx
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
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
    
    def predict(self, x, sub_df):
        preds = np.zeros((len(sub_df), 4))
        for cv_idx in range(self.cv_num):
            model = tf.keras.models.load_model('model/' + self.name + '_{0}'.format(cv_idx))
            preds += model(x).numpy()
            if self.sample_num != 0:
                for sample in range(self.sample_num - 1):
                    preds += model(x).numpy()
                preds /= self.sample_num
        sub_df.iloc[:, 1] = np.argmax(preds, axis=1) + 1
        sub_df.to_csv('submit/' + self.name + '.csv', index=None, header=None)
    
    def run(self):
        train_df, test_df, sub_df = self.read_data()
        train, test, vocab_size = self.preprocess(train_df, test_df)
        
        if self.is_train:
            loss = self.cv(train, train_df['jobflag'].values - 1, vocab_size + 1)
            print('cv test:{0:.4f}'.format(np.mean(loss)))

        self.predict(test, sub_df)

        
if __name__ == '__main__':
    stream = Stream(Baseline, 28, 50, 128, 'baseline')
    stream.run()
