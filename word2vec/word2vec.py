from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import json
import os

class DataLoader(tf.keras.utils.Sequence):

    """Lazy loading word positive and negative pairs"""

    def __init__(self, word_ids, tokens, batch_size=1024):
        self.word_ids = word_ids
        self.tokens = tokens
        self.batch_size = batch_size
        self.indices = np.arange(len(tokens))


    def __len__(self):
        return len(self.tokens)//self.batch_size

    def __getitem__(self, batch_index):
        batch_indices = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        selected_tokens = self.tokens[batch_indices]
        selected_tokens = np.pad(selected_tokens, [(0,0), (0, 1)])

        # Label selections
        labels = np.ones((self.batch_size, 1), dtype=np.int32)#np.random.randint(2, size=self.batch_size)

        # Ensure we do not take any padding characters
        first_zeros = (selected_tokens==0).astype(int).argmax(axis=-1).reshape((self.batch_size, 1))
        source_indices = (np.random.random((self.batch_size, 1)) * first_zeros.reshape((self.batch_size, 1))).astype(int).reshape((self.batch_size, 1))
        source_final = np.array([selected_tokens[x[0] , x[1] - 1] for x in np.concatenate((np.arange(self.batch_size).reshape((self.batch_size, 1)), source_indices), axis=-1)]).reshape((self.batch_size, 1))
        targets_final =  np.array([selected_tokens[x[0], x[1]] for x in np.concatenate((np.arange(self.batch_size).reshape((self.batch_size, 1)), source_indices), axis=-1)]).reshape((self.batch_size, 1))

        random_words = np.random.randint(len(self.word_ids) - 1, size=self.batch_size//2) + 1
        labels[:self.batch_size//2, 0] = 0
        for i in range(self.batch_size//2):
            targets_final[i, 0] = random_words[i]
        return [source_final, targets_final], labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class Word2VecModel():
    def __init__(self, projection_dim=128):
        self.word_ids = {}
        self.precomputed = None
        self.projection_dim = projection_dim

    def _pad(self, tokens, length=10):
        result = []
        for i in range(len(tokens)):
            row = ['pad'] * 10
            for j in range(min(len(tokens[i]), length)):
                row[j] = tokens[i][j]
            result.append(row)

        return np.array(result)

    def predict(self, word):

        if self.precomputed is None:
            keys = [x for x in self.word_ids.keys()]
            predictions = self.model.predict([self.word_ids[x] for x in keys])
            self.precomputed = {keys[i]:predictions[i] for i in range(len(keys))}
        if word in self.precomputed:
            return self.precomputed[word]
        else:
            return np.zeros(self.projection_dim)

    def save(self, path):
        self.model.save(path)
        with open(os.path.join(path, "ids.json"), "w+") as f:
            json.dump(self.word_ids, f)
        with open(os.path.join(path, "other_configs.json"), "w+") as f:
            json.dump({"projection_dim": self.projection_dim}, f)
        
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        with open(os.path.join(path, "ids.json"), "r") as f:
            ids = json.load(f)
        with open(os.path.join(path, "other_configs.json"), "r") as f:
            self.projection_dim = int(json.load(f)["projection_dim"])
        for key in ids.keys():
            ids[key] = int(ids[key])
        self.word_ids = ids



    def fit(self, tokens, epochs=32, batch_size=1024):
        '''
        @param tokens: A 2d array. Each row contains a set of words from an input description
        '''

        tokens = self._pad(tokens)

        # Conver the uniue words in the input to a lookup dict
        unique = np.unique(tokens.flatten())
        num_unique = len(unique)
        self.word_ids = {'pad': 0}
        insert_index = 1
        for i in range(num_unique):
             if unique[i] != 'pad':
                 self.word_ids[unique[i]] = insert_index
                 insert_index += 1
        tokens = np.array([self.word_ids[x] for x in tokens.flatten()]).reshape(tokens.shape)

        # Make training model
        embedding_layer =  layers.Embedding(input_dim=num_unique, output_dim=self.projection_dim)

        # Inputs
        tokens_1 = layers.Input(shape=(1,))
        tokens_2 = layers.Input(shape=(1, ))
        
        # Get embeddings
        vec_1 = embedding_layer(tokens_1)
        vec_2 = embedding_layer(tokens_2)

        # Scale dot products
        dot = layers.Dot(axes=-1)([vec_1, vec_2])
        dot = layers.Activation(activation='sigmoid')(dot)
        training_model = Model([tokens_1, tokens_2], dot)

        # Make model that will be used for actual predictions
        real_inputs = layers.Input(shape=(1,))
        out = embedding_layer(real_inputs)
        self.model = Model(real_inputs, out)

        # Run training
        data_loader = DataLoader(self.word_ids, tokens, batch_size=batch_size)
        training_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        training_model.fit(x=data_loader, epochs=epochs, verbose=2)




