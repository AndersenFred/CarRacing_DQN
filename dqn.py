import tensorflow as tf
import gym
class DQNAgent:
    def __init__(self, inputshape, outputsize, memory_length, batchsize = 128):
        self.memory_length = memory_length
        self.batchsize = batchsize
        self.model = self.create_agent(inputshape, outputsize)
        self.memory = []
    def create_agent(self, inputshape, outputsize):

        inputs = tf.keras.Input(inputshape)
        conf1 = tf.keras.layers.Conv2D(4*inputshape[-1],7)(inputs)
        relu1 = tf.keras.layers.LeakyReLU()(conf1)
        norm1 = tf.keras.layers.BatchNormalization()(relu1)

        conf2 = tf.keras.layers.Conv2D(8*inputshape[-1],5)(norm1)
        relu2 = tf.keras.layers.LeakyReLU()(conf2)
        norm2 = tf.keras.layers.BatchNormalization()(relu2)

        conf3 = tf.keras.layers.Conv2D(16 * inputshape[-1], 5, activation = 'relu')(norm2)
        relu3 = tf.keras.layers.LeakyReLU()(conf3)
        norm3 = tf.keras.layers.BatchNormalization()(relu3)

        conf4 = tf.keras.layers.Conv2D(16 * inputshape[-1], 3, activation = 'relu')(norm3)
        relu4 = tf.keras.layers.LeakyReLU()(conf4)
        norm4 = tf.keras.layers.BatchNormalization()(relu4)

        conf5 = tf.keras.layers.Conv2D(16 * inputshape[-1], 3, activation='relu')(norm4)
        relu5 = tf.keras.layers.LeakyReLU()(conf5)
        norm5 = tf.keras.layers.BatchNormalization()(relu5)

        flat = tf.keras.layers.Flatten()(norm5)

        fc1 = tf.keras.layers.Dense(512)(flat)
        relu6 = tf.keras.layers.LeakyReLU()(fc1)

        fc2 = tf.keras.layers.Dense(64)(relu6)
        relu7 = tf.keras.layers.LeakyReLU()(fc2)
        out =tf.keras.layers.Dense(outputsize, activation='sigmoid')(relu7)

        return tf.keras.Model(inputs=inputs, outputs = out)

DQNAgent((96,96,3),5,512).model.summary()
