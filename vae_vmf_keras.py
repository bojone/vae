#! -*- coding: utf-8 -*-
# vMF-VAE简单实现参考

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist

# 基本参数
batch_size = 100
original_dim = 784
latent_dim = 4
intermediate_dim = 256
epochs = 50
kappa = 20

# 加载数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 模型定义
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 参数mu
mu = Dense(latent_dim)(h)
mu = Lambda(lambda x: K.l2_normalize(x, axis=-1))(mu)


def sampling(mu):
    """vMF分布重参数操作
    """
    dims = K.int_shape(mu)[-1]
    # 预先计算一批w
    epsilon = 1e-7
    x = np.arange(-1 + epsilon, 1, epsilon)
    y = kappa * x + np.log(1 - x**2) * (dims - 3) / 2
    y = np.cumsum(np.exp(y - y.max()))
    y = y / y[-1]
    W = K.constant(np.interp(np.random.random(10**6), y, x))
    # 实时采样w
    idx = K.random_uniform(K.shape(mu[:, :1]), 0, 10**6, dtype='int32')
    w = K.gather(W, idx)
    # 实时采样z
    eps = K.random_normal(K.shape(mu))
    nu = eps - K.sum(eps * mu, axis=1, keepdims=True) * mu
    nu = K.l2_normalize(nu, axis=-1)
    return w * mu + (1 - w**2)**0.5 * nu


# 重参数层
z = Lambda(sampling)(mu)

# 解码层
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
vae = Model(x, x_decoded_mean)

loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
vae.add_loss(K.mean(loss))
vae.compile(optimizer='adam')
vae.summary()

vae.fit(
    x_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None)
)

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 观察随机采样结果
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# 球面上均匀采样
for i in range(n):
    for j in range(n):
        z_sample = np.random.randn(1, latent_dim)
        z_sample /= (z_sample**2).sum()**0.5
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size:(i + 1) * digit_size,
               j * digit_size:(j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.savefig('test.png')
