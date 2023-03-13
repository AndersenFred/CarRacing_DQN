import tensorflow as tf
import gc
import numpy as np
import random
class DQNAgent:

    def __init__(self, inputshape, outputsize, capacity, batchsize:int = 32, epsilon = 0.05, lr = 1e-3, gamma = 0.9):
        if not (type(batchsize) == int):
            raise ValueError(f'bachsize must be type of in: type = {type(batchsize)}')
        if not (type(capacity) == int):
            raise ValueError(f'capacity must be type of in: type = {type(capacity)}')
        if capacity < batchsize:
            raise ValueError(f'capacity has to be greater than batchsize: capacity = {capacity}, batchsize = {batchsize}')
        #tf.config.set_visible_devices([], 'GPU')
        self.capacity = capacity
        self.lr = lr
        self.batchsize = batchsize
        self.model = self.create_agent(inputshape, outputsize)
        self.target_model = self.create_agent(inputshape, outputsize)
        self.memory = []
        self.epsilon = epsilon
        self.outputsize = outputsize
        self.gamma = gamma
        self.model.summary()
        gc.collect()

    def create_agent(self, inputshape, outputsize):

        inputs = tf.keras.Input(inputshape)
        conf1 = tf.keras.layers.Conv2D(2*inputshape[-1], 7, strides=3)(inputs)
        relu1 = tf.keras.layers.LeakyReLU()(conf1)

        conf2 = tf.keras.layers.Conv2D(4*inputshape[-1], 5, strides=2)(relu1)
        relu2 = tf.keras.layers.LeakyReLU()(conf2)


        conf3 = tf.keras.layers.Conv2D(8 * inputshape[-1], 3)(relu2)
        relu3 = tf.keras.layers.LeakyReLU()(conf3)

        flat = tf.keras.layers.Flatten()(relu3)

        fc2 = tf.keras.layers.Dense(512)(flat)
        relu7 = tf.keras.layers.LeakyReLU()(fc2)
        out = tf.keras.layers.Dense(outputsize, activation='sigmoid')(relu7)

        model = tf.keras.Model(inputs=inputs, outputs = out)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-7))
        return model

    def add_to_memory(self, state, action, next, reward, done):
        self.memory.append((state, action, next, reward, done))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def get_random_sample(self):
        if len(self.memory) < self.batchsize:
            return None
        return random.sample(self.memory, self.batchsize)

    def act(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.model.predict(np.expand_dims(state, axis=0), verbose = 0))
        return np.random.randint(0, self.outputsize)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        if len(name) < 4 or not (name[-3:] == '.h5' or name[-3:] == '.tf'):
            name = name+'.h5'
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.models.load_model(name)
        self.update_target_model()

    def train(self):
        batch = self.get_random_sample()
        if batch == None:
            return
        train_state = []
        train_target = []
        for state, action, next, reward, done in batch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose = 0, batch_size= self.batchsize)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next, axis=0), verbose = 0, batch_size= self.batchsize)[0]
                target[action] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0, batch_size= self.batchsize)





