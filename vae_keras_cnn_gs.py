#! -*- coding: utf-8 -*-

'''用Keras实现的VAE，CNN版本
   使用了离散隐变量，为此使用了Gumbel Softmax做重参数。
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
'''

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import Layer
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import Callback


# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# 网络参数
input_shape = (image_size, image_size, 1)
batch_size = 100
kernel_size = 3
filters = 16
num_latents = 32
classes_per_latent = 10 # 这里假设隐变量是num_latents维、classes_per_latent元随机变量
epochs = 30


x_in = Input(shape=input_shape)
x = x_in

for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# 备份当前shape，等下构建decoder的时候要用
shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)
logits = Dense(num_latents * classes_per_latent)(x)
logits = Reshape((num_latents, classes_per_latent))(logits)

class GumbelSoftmax(Layer):
    """Gumbel Softmax重参数
    """
    def __init__(self, tau=1., **kwargs):
        super(GumbelSoftmax, self).__init__(**kwargs)
        self.tau = K.variable(tau)
    def call(self, inputs):
        epsilon = K.random_uniform(shape=K.shape(inputs))
        epsilon = - K.log(epsilon + K.epsilon())
        epsilon = - K.log(epsilon + K.epsilon())
        outputs = inputs + epsilon
        outputs = K.softmax(outputs / self.tau, -1)
        return outputs

gumbel_softmax = GumbelSoftmax()
z_sample = gumbel_softmax(logits)

# 解码层，也就是生成器部分
# 先搭建为一个独立的模型，然后再调用模型
latent_inputs = Input(shape=(num_latents, classes_per_latent))
x = Reshape((num_latents * classes_per_latent,))(latent_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same')(x)

# 搭建为一个独立的模型
decoder = Model(latent_inputs, outputs)

x_out = decoder(z_sample)

# 建立模型
vae = Model(x_in, x_out)

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x_in, x_out), axis=[1, 2, 3])
p = K.clip(K.softmax(logits, -1), K.epsilon(), 1 - K.epsilon())
# 假设先验分布为均匀分布，那么kl项简化为负熵
kl_loss = K.sum(p * K.log(p), axis=[1, 2])
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

class Trainer(Callback):
    def __init__(self):
        self.max_tau = 1.
        self.min_tau = 0.01
        self._tau = self.max_tau - self.min_tau
    def on_batch_begin(self, batch, logs=None):
        tau = self.min_tau + self._tau
        K.set_value(gumbel_softmax.tau, tau)
        self._tau *= 0.999
    def on_epoch_begin(self, epoch, logs=None):
        tau = K.eval(gumbel_softmax.tau)
        print('epoch: %s, tau: %.5f' % (epoch + 1, tau))

trainer = Trainer()
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
        callbacks=[trainer])


# 观察隐变量的两个维度变化是如何影响输出结果的
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        z_sample = np.zeros((1, num_latents, classes_per_latent))
        for iz in range(num_latents):
            jz = np.random.choice(classes_per_latent)
            z_sample[0, iz, jz] = 1
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
