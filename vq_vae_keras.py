#! -*- coding: utf-8 -*-
# Keras简单实现VQ-VAE

import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
import os


if not os.path.exists('samples'):
    os.mkdir('samples')


imgs = glob.glob('../../CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 128
num_codes = 64
batch_size = 64
num_layers = int(np.log2(img_dim) - 4)


def imread(f):
    x = misc.imread(f, mode='RGB')
    x = misc.imresize(x, (img_dim, img_dim))
    x = x.astype(np.float32)
    return x / 255 * 2 - 1


class img_generator:
    """图片迭代器，方便重复调用
    """
    def __init__(self, imgs, batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i,f in enumerate(self.imgs):
                X.append(imread(f))
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    yield X, None
                    X = []


def resnet_block(x):
    """残差块
    """
    dim = K.int_shape(x)[-1]
    xo = x
    x = Activation('relu')(x)
    x = Conv2D(dim, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(dim, 1, padding='same')(x)
    return Add()([xo, x])


# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

x = Conv2D(z_dim, 4, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(z_dim, 4, strides=2, padding='same')(x)
x = BatchNormalization()(x)

for i in range(num_layers):
    x = resnet_block(x)
    if i < num_layers - 1:
        x = BatchNormalization()(x)

e_model = Model(x_in, x)
e_model.summary()


# 解码器
z_in = Input(shape=K.int_shape(x)[1:])
z = z_in

for i in range(num_layers):
    z = BatchNormalization()(z)
    z = resnet_block(z)

z = Conv2DTranspose(z_dim, 4, strides=2, padding='same')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, 4, strides=2, padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 硬编码模型
z_in = Input(shape=K.int_shape(x)[1:])
z = z_in

class VectorQuantizer(Layer):
    """量化层
    """
    def __init__(self, num_codes, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_codes = num_codes
    def build(self, input_shape):
        super(VectorQuantizer, self).build(input_shape)
        dim = input_shape[-1]
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.num_codes, dim),
            initializer='uniform'
        )
    def call(self, inputs):
        """inputs.shape=[None, m, m, dim]
        """
        l2_inputs = K.sum(inputs**2, -1, keepdims=True)
        l2_embeddings = K.sum(self.embeddings**2, -1)
        for _ in range(3):
            l2_embeddings = K.expand_dims(l2_embeddings, 0)
        embeddings = K.transpose(self.embeddings)
        dot = K.dot(inputs, embeddings)
        distance = l2_inputs + l2_embeddings - 2 * dot
        codes = K.cast(K.argmin(distance, -1), 'int32')
        code_vecs = K.gather(self.embeddings, codes)
        return [codes, code_vecs]
    def compute_output_shape(self, input_shape):
        return [input_shape[:-1], input_shape]

vq_layer = VectorQuantizer(num_codes)
codes, code_vecs = vq_layer(z)

q_model = Model(z_in, [codes, code_vecs])
q_model.summary()


# 训练模型
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

z = e_model(x)
_, e = q_model(z)
ze = Lambda(lambda x: x[0] + K.stop_gradient(x[1] - x[0]))([z, e])
x = g_model(ze)

train_model = Model(x_in, [x, _])

mse_x = K.mean((x_in - x)**2)
mse_e = K.mean((K.stop_gradient(z) - e)**2)
mse_z = K.mean((K.stop_gradient(e) - z)**2)
loss = mse_x + mse_e + 0.25 * mse_z

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()
train_model.metrics_names.append('mse_x')
train_model.metrics_tensors.append(mse_x)
train_model.metrics_names.append('mse_e')
train_model.metrics_tensors.append(mse_e)
train_model.metrics_names.append('mse_z')
train_model.metrics_tensors.append(mse_z)


# 重构采样函数
def sample_ae_1(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


# 重构采样函数
def sample_ae_2(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                z_sample = q_model.predict(z_sample)[1]
                x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


# 随机线性插值
def sample_inter(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        img1, img2 = np.random.choice(imgs, 2)
        z_sample_1, z_sample_2 = e_model.predict(np.array([imread(img1), imread(img2)]))
        z_sample_1, z_sample_2 = np.array([z_sample_1]), np.array([z_sample_2])
        for j in range(n):
            alpha = j / (n - 1.)
            z_sample = (1 - alpha) * z_sample_1 + alpha * z_sample_2
            z_sample = q_model.predict(z_sample)[1]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100
    def on_batch_end(self, batch, logs=None):
        if self.batch % self.iters_per_sample == 0:
            sample_ae_1('samples/test_ae_1_%s.png' % self.batch)
            sample_ae_2('samples/test_ae_2_%s.png' % self.batch)
            train_model.save_weights('./train_model.weights')
        self.batch += 1
        batch = min(self.batch, 100000.)


if __name__ == '__main__':

    trainer = Trainer()
    img_data = img_generator(imgs, batch_size)

    train_model.fit_generator(img_data.__iter__(),
                              steps_per_epoch=len(img_data),
                              epochs=1000,
                              callbacks=[trainer])

