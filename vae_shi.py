#! -*- coding:utf-8 -*-
# 一个简单的基于VAE和CNN的作诗机器人
# 来自：https://kexue.fm/archives/5332

import re
import codecs
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback


n = 5 # 只抽取五言诗
latent_dim = 64 # 隐变量维度
hidden_dim = 64 # 隐层节点数

s = codecs.open('shi.txt', encoding='utf-8').read()

# 通过正则表达式找出所有的五言诗
s = re.findall(u'　　(.{%s}，.{%s}。.*?)\r\n'%(n,n), s)
shi = []
for i in s:
    for j in i.split(u'。'): # 按句切分
        if j:
            shi.append(j)

shi = [i[:n] + i[n+1:] for i in shi if len(i) == 2*n+1]

# 构建字与id的相互映射
id2char = dict(enumerate(set(''.join(shi))))
char2id = {j:i for i,j in id2char.items()}

# 诗歌id化
shi2id = [[char2id[j] for j in i] for i in shi]
shi2id = np.array(shi2id)


class GCNN(Layer): # 定义GCNN层，结合残差
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                     shape=(3, input_shape[-1],
                                            self.output_dim * 2),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, x):
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:,:,:self.output_dim] * K.sigmoid(_[:,:,self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _


input_sentence = Input(shape=(2*n,), dtype='int32')
input_vec = Embedding(len(char2id), hidden_dim)(input_sentence) # id转向量
h = GCNN(residual=True)(input_vec) # GCNN层
h = GCNN(residual=True)(h) # GCNN层
h = GlobalAveragePooling1D()(h) # 池化

# 算均值方差
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# 定义解码层，分开定义是为了后面的重用
decoder_hidden = Dense(hidden_dim*(2*n))
decoder_cnn = GCNN(residual=True)
decoder_dense = Dense(len(char2id), activation='softmax')

h = decoder_hidden(z)
h = Reshape((2*n, hidden_dim))(h)
h = decoder_cnn(h)
output = decoder_dense(h)


# 建立模型
vae = Model(input_sentence, output)

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# 重用解码层，构建单独的生成模型
decoder_input = Input(shape=(latent_dim,))
_ = decoder_hidden(decoder_input)
_ = Reshape((2*n, hidden_dim))(_)
_ = decoder_cnn(_)
_output = decoder_dense(_)
generator = Model(decoder_input, _output)


# 利用生成模型随机生成一首诗
def gen():
    r = generator.predict(np.random.randn(1, latent_dim))[0]
    r = r.argmax(axis=1)
    return ''.join([id2char[i] for i in r[:n]])\
           + u'，'\
           + ''.join([id2char[i] for i in r[n:]])


# 回调器，方便在训练过程中输出
class Evaluate(Callback):
    def __init__(self):
        self.log = []
    def on_epoch_end(self, epoch, logs=None):
        self.log.append(gen())
        print (u'          %s'%(self.log[-1])).encode('utf-8')


evaluator = Evaluate()

vae.fit(shi2id,
        shuffle=True,
        epochs=100,
        batch_size=64,
        callbacks=[evaluator])

vae.save_weights('shi.model')

for i in range(20):
    print gen()
