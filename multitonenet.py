import numpy as np
from sklearn import metrics
import os
import sys
import shutil
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.constraints import unit_norm
from keras.utils import np_utils
from keras.layers import TimeDistributed,ZeroPadding1D,Masking, Reshape, Conv2D,Conv1D, Activation, MaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, Dense, GRU,Bidirectional,multiply, subtract, concatenate, add, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.engine.input_layer import Input
from keras.models import Model 
from keras.utils import Sequence
from keras.optimizers import Adam,SGD,Nadam
from adabound import AdaBound
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from wers import *
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerACT, TransformerBlock
from keras_self_attention import SeqSelfAttention
from tensorflow.python.ops import math_ops as tf_math_ops
from tensorflow.python.ops import ctc_ops 
from itertools import groupby
from layer_normalization import LayerNormalization
from keras.initializers import TruncatedNormal
from keras_pos_embd import TrigPosEmbedding
from transformer import SelfAttention, AddPosEncoding
# from flip_gradient import GradientReversal

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
# 设置session
K.set_session(session)

class OutputModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(OutputModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(OutputModelCheckpoint,self).set_model(self.single_model)

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda=1.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LR_WarmUP_Exponential_Decay(Callback):
    def __init__(self, lr, warmup_epochs, decay_k=1):
        self.num_passed_batchs = 0
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.decay_k = decay_k # 指数衰减超参
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.params['steps'] == None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            # 前10个epoch中，学习率线性地从零增加到0.001
            lr_now = self.lr * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs
            K.set_value(self.model.optimizer.lr, lr_now)
        else:
            # 10个epoch后，学习率线开始指数衰减
            lr_now = self.lr * (self.decay_k / (self.num_passed_batchs / self.steps_per_epoch) ** 0.5)
            K.set_value(self.model.optimizer.lr, lr_now)

        sys.stdout.write(' '*100+'\r') # 先用空格清屏
        sys.stdout.write("LR = "+str(lr_now)+'\r')

        self.num_passed_batchs += 1
    # def on_epoch_end(sef, epoch, logs={}):
    #     sys.stdout.write('\n')
        
class LR_Segment_Exponential_Decay(Callback):
    def __init__(self, lr, segment_epochs, decay_k=1):
        self.num_passed_batchs = 0
        self.segment_epochs = segment_epochs
        self.lr = lr
        self.decay_k = decay_k # 指数衰减超参
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.params['steps'] == None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.segment_epochs:
            lr_now = self.lr #/ 2.0
            #K.set_value(self.model.optimizer.lr, lr_now)
        else:
            # 学习率线开始指数衰减
            history_epochs = self.segment_epochs*self.steps_per_epoch
            lr_now = self.lr * (self.decay_k / ((self.num_passed_batchs-history_epochs+1) / self.steps_per_epoch) ** 0.5)
            K.set_value(self.model.optimizer.lr, lr_now)

        sys.stdout.write(' '*200+'\r') # 先用空格清屏
        sys.stdout.write("LR = "+str(lr_now)+'\r')

        self.num_passed_batchs += 1
    # def on_epoch_end(sef, epoch, logs={}):
    #     sys.stdout.write('\n')

class MetricCallback(Callback):
    
    def __init__(self, test_func, x, y, input_length, batch_size=1, test_num=32,info='this is test'):
        self.test_func = test_func
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.test_num = test_num if test_num != None else len(self.x)
        self.input_length=input_length
        self.info = info
        self.ctc_class = Ctc_Decode(batch_size=None,timestep=None, nclass=6)

    def on_epoch_end(self, epoch, logs={}):
        print('\n'+self.info)

        y_pred = []
        prob = []
        for i in range(self.test_num):
            y_pred_tmp, prob_tmp = decode_batch(self.test_func, np.array([self.x[i]]),[self.input_length[i]],self.ctc_class)
            y_pred.extend(y_pred_tmp)
            prob.extend(prob_tmp)

        y_pred = [''.join([str(int(p)).replace('-1', '') for p in pred]) for pred in y_pred]
        y_true = [''.join([str(t) for t in true]) for true in self.y[0:self.test_num]]

        with open('tmp5', 'a', encoding='utf-8') as f:
            f.write( str(list(zip(y_pred, y_true))) + '\n' )
        
        _, mean = wers(y_true, y_pred)
        print('WER(%s):'%(self.test_num), mean,'\n')

        del y_pred_tmp
        del prob_tmp


def get_data(data_path_file, root, stride=4, test_num=1024):
    with open(data_path_file,'r',encoding='utf-8') as f:
        data = f.readlines()
    random.shuffle(data)
    content=[]
    label = []#tone type
    input_length = []
    #tone2index = {" ":0,"0":1,"1":2,"2":3,"3":4,"4":5}
    if test_num != None:
        data = data[:test_num]
    for d in data:#[:test_num]:
        tmp = d.split('\t')
        x = lowf_feature(os.path.join(root, tmp[0]+'.wav'), save_path='/data/gq/sjc/Version_3/data1')
        
        content.append(x)
        input_length.append(math.ceil(x.shape[0]/stride))#-2)

        label_tmp = [int(i) for i in tmp[1].split(' ')]
        label.append(label_tmp)

    # max_audio_lens = max([i.shape[0] for i in content])
    # for i in range(test_num):
    #     if content[i].shape[0] < max_audio_lens:
    #         zero = np.zeros((max_audio_lens-content[i].shape[0],content[i].shape[1]),dtype=np.float32)
    #         content[i] = np.vstack((content[i],zero))
    return content, label, input_length

def lowf_feature(audio_path, save_path):
    audio_root = os.path.dirname(audio_path)
    save_path_f = os.path.join(save_path, '/'.join(audio_root.split('/')[-2:]))
    name = os.path.basename(audio_path).split('.')[0] + '.npy'

    if not os.path.exists(save_path_f):
        os.makedirs(save_path_f)

    path = os.path.join(save_path_f, name)
    if os.path.isfile(path) and os.path.getsize(path) > 256:
        x = np.load(path)
    else:
        # x = melspectrogram_feature(audio_path, save_path_f)
        x = extract_features_shennong(audio_path, save_path_f)
    return x#np.expand_dims(x, axis=-1)

from shennong.audio import Audio
from shennong.features.processor.filterbank import FilterbankProcessor
from shennong.features.processor.pitch import (PitchProcessor, PitchPostProcessor)
def extract_features_shennong(audio_path, save_path):
    audio = Audio.load(audio_path)
    # 80-dim fbank with 1-dim energe
    processor = FilterbankProcessor(sample_rate=audio.sample_rate, num_bins=40, use_energy=False)#80 fbank + 1 energy
    fbank = processor.process(audio)
    fbank = fbank.data#(fbank.data - fbank.data.mean()) / fbank.data.std()

    # 3-dim pitch
    processor = PitchProcessor(frame_shift=0.01, frame_length=0.025)
    options = {
                'sample_rate': audio.sample_rate,
                'frame_shift': 0.01, 'frame_length': 0.025,
                'min_f0': 20, 'max_f0': 500}
    processor = PitchProcessor(**options)
    pitch = processor.process(audio)
    postprocessor = PitchPostProcessor()  # use default options
    postpitch = postprocessor.process(pitch) # 3 dim
    postpitch = postpitch.data#(postpitch.data - postpitch.data.mean()) / postpitch.data.std()
    #features = postpitch
    shape = min(fbank.shape[0], postpitch.shape[0])
    #zero = np.zeros((,content[i].shape[1]),dtype=np.float32)
    #content[i] = np.vstack((content[i],zero))
    features = np.concatenate((fbank[:shape,:], postpitch[:shape,:]), axis=-1)

    # name = os.path.basename(audio_path).split('.')[0] + '.npy'
    # np.save(os.path.join(save_path, name), features.data)
    return features

        
import random
import math
class Generate_arrays_from_file(Sequence):
    def __init__(self, data_path_file, batch_size, stride=4, flag='train'):
        # 初始化所需的参数
        self.data_path_file = data_path_file
        self.batch_size = batch_size
        self.flag = flag
        self.stride = stride

    def __len__(self):
        # 让代码知道这个序列的长度
        with open(self.data_path_file,'r',encoding='utf-8') as f:
            self.data = f.readlines()
        random.shuffle(self.data)
        self.all_num_epoch = math.ceil(len(self.data)/self.batch_size)
        return self.all_num_epoch

    # def on_epoch_end(self, epoch='per epoch end', logs={}):
    #    print('shuffle data')
    #    random.shuffle(self.data)

    def __getitem__(self, idx):
        # if idx == 0:
        #     print('shuffle data')
        #     random.shuffle(self.data)
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        random.shuffle(batch_x)
        # 迭代器部分
        content = []
        label = []#tone type
        input_length=[]#某个特征输入的长度
        label_length = []#label lens
        label_gender = []
        train_root = '/data/gq/sjc/aishell-1/data_aishell/wav/' + self.flag
        #tone2index = {" ":0,"0":1,"1":2,"2":3,"3":4,"4":5}
        gender2index = {'M':0,'F':1}
        for d in batch_x:
            audio_name = d.split('\t')[0]
            audio = os.path.join(train_root, audio_name+'.wav')
            x = lowf_feature(audio, save_path='/data/gq/sjc/Version_3/data1')
            # the 2 is critical here since the first couple outputs of the RNN
            # tend to be garbage:
            input_length.append(np.array([math.ceil(x.shape[0]/self.stride)]))#-2)
            content.append(x)

            label_tmp = [int(i) for i in d.split('\t')[1].split(' ')]
            label_length.append(np.array([len(label_tmp)]))
            label.append(label_tmp)

            label_gender.append(gender2index[d.split('\t')[2].strip('\n')])

        # with open('tmp6','a',encoding='utf-8') as f:
        #     f.write(str(self.batch_size)+'\t'+str(len(content))+'\t'+str(len(label_f))+'\t'+str(len(label_n))+'\n')
        # padding audio 长度对齐
        max_audio_lens = max([i.shape[0] for i in content])
        # content = pad_sequences(content, maxlen=max_audio_lens, dtype='float32', padding='post', truncating='post', value=0.0)
        for i in range(len(content)):
            try:
                if content[i].shape[0] < max_audio_lens:
                    # with open('tmp','a',encoding='utf-8') as f:
                    #     f.write(str(content[i].shape)+'\t')
                    zero = np.zeros((max_audio_lens-content[i].shape[0],content[i].shape[1]),dtype=np.float32)
                    content[i] = np.vstack((content[i],zero))
            except:
                print(i,111111111)
                print(len(content),222222222)
                print(batch_x[i],33333333)
                print(content[i].shape,44444444)

            # with open('tmp','a',encoding='utf-8') as f:
            #     f.write(str(max_audio_lens)+'\t'+str(content[i].shape)+'\n')

        # padding text 长度对齐
        max_text_lens = max([len(i) for i in label])
        # label = pad_sequences(label, maxlen=max_text_lens, dtype='float32', padding='post', truncating='post', value=0.0)
        for i in range(len(label)):
            if len(label[i]) < max_text_lens:
                # with open('tmp2','a',encoding='utf-8') as f:
                #     f.write(str(len(label[i]))+'\t')
                zero = []
                for j in range(max_text_lens-len(label[i])):
                    zero.append(-1) # 5 is the space
                label[i].extend(zero)
            label[i] = np.array(label[i])
            # with open('tmp2','a',encoding='utf-8') as f:
            #     f.write(str(max_text_lens)+'\t'+str(label[i].shape)+'\n')
     
        #del zero
        #del max_audio_lens
        #del max_text_lens

        inputs = {'the_inputs': np.array(content,dtype=np.float32),
                  'the_labels': np.array(label,dtype=np.float32),
                  'input_length': np.array(input_length,dtype=np.int32),
                  'label_length': np.array(label_length,dtype=np.int32)}

        outputs = {'ctc': np.zeros([len(label)]), 'gender': np_utils.to_categorical(label_gender,num_classes = 2)} # dummy data for dummy loss function

        return inputs, outputs

def resblock(cnn_input, f, k, activation, size,num_heads,batch_size):

    cnn_in = Conv1D(filters=f//2,kernel_size=1,strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input)
    cnn_in=BatchNormalization()(cnn_in)
    cnn_in=GELU(cnn_in)

    cnn = muti_channel_sa(cnn_in, 'softmax', batch_size, attention_width=None,num_channels=num_heads)
    cnn=BatchNormalization()(cnn)
    cnn=GELU(cnn)

    cnn = concatenate([cnn,cnn_in],-1)

    cnn = Conv1D(filters=f,kernel_size=1,strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn)
    cnn=BatchNormalization()(cnn)

    cnn = add([cnn,cnn_input])

    cnn=GELU(cnn)

    return cnn

def muti_channel_sa(cnn_input, activation, batch_size, attention_width=None, num_channels=8):
    input_shape = cnn_input.shape
    if attention_width != None:
        attention_width = min(attention_width, input_shape[1])
    if num_channels == 1:
        cnn_output = SeqSelfAttention(attention_width=attention_width,attention_activation=activation)(cnn_input)
    else:
        cnn_output=[]
        for nh in range(num_channels):
            cnn_output.append(SeqSelfAttention(attention_width=attention_width,attention_activation=activation)(cnn_input))
        cnn_output = concatenate(cnn_output,-1)

    cnn_output = TimeDistributed(Dropout(0.1))(cnn_output)
    return cnn_output

def GELU(cnn_input):
    flag=3
    if flag == 1:
        cdf = Lambda(lambda x: 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0))))(cnn_input) 
        return multiply([cnn_input, cdf])
    elif flag == 2:
        return Mish(cnn_input)
    elif flag == 3:
        return Activation('relu')(cnn_input)

def Mish(cnn_input):
    return Lambda(lambda x: x * tf.tanh(tf.nn.softplus(x)))(cnn_input) 

def blur_pool_func(x, filt_size=3, stride=2):
    stride = (stride,stride)
    padding = ( (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ), (int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)) ) )
    if(filt_size==1):
        k = np.array([1.,])
    elif(filt_size==2):
        k = np.array([1., 1.])
    elif(filt_size==3):
        k = np.array([1., 2., 1.])
    elif(filt_size==4):    
        k = np.array([1., 3., 3., 1.])
    elif(filt_size==5):    
        k = np.array([1., 4., 6., 4., 1.])
    elif(filt_size==6):    
        k = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size==7):    
        k = np.array([1., 6., 15., 20., 15., 6., 1.])
    #k = a
    k = k[:,None]*k[None,:]
    k = k / np.sum(k)
    k = np.tile (k[:,:,None,None], (1,1,K.int_shape(x)[-1],1) )                
    k = K.constant (k, dtype=K.floatx() )
    
    x = K.spatial_2d_padding(x, padding=padding)
    x = K.depthwise_conv2d(x, k, strides=stride, padding='valid')
    return x
def blur_pool(cnn_input, filt_size=3, stride=2):
    #return blur_pool_func(cnn_input, filt_size=filt_size, stride=stride)
    return Lambda(lambda x: blur_pool_func(x, filt_size=filt_size, stride=stride))(cnn_input) 

# 被creatModel调用，用作ctc损失的计算
def ctc_loss_lambda(args):
    labels, y_pred, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def muti_scale_cnn2d(cnn_input_2d, filters_2d, strides, mode='add'):
   
    cnn2d = Conv2D(filters=filters_2d,kernel_size=[1,3],strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_2d)
    cnn2d=BatchNormalization()(cnn2d)
    cnn2d_1=GELU(cnn2d)
    
    cnn2d = Conv2D(filters=filters_2d,kernel_size=[3,1],strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_2d)
    cnn2d=BatchNormalization()(cnn2d)
    cnn2d_2=GELU(cnn2d)

    cnn2d = Conv2D(filters=filters_2d,kernel_size=[3,3],strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_2d)
    cnn2d=BatchNormalization()(cnn2d)
    cnn2d_3=GELU(cnn2d)

    if mode== 'add':
        cnn2d = Conv2D(filters=filters_2d,kernel_size=[1,1],strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_2d)
        cnn2d=BatchNormalization()(cnn2d)
        cnn2d=GELU(cnn2d)
        cnn2d = add([cnn2d_1, cnn2d_2, cnn2d_3, cnn2d])
    else:
        cnn2d = concatenate([cnn2d_1, cnn2d_2,cnn2d_3, cnn_input_2d],-1)

    cnn2d = Conv2D(filters=filters_2d,kernel_size=[5,5],strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn2d)
    cnn2d=BatchNormalization()(cnn2d)
    cnn2d=GELU(cnn2d)

    cnn2d=blur_pool(cnn2d, filt_size=5, stride=strides)

    return cnn2d

def muti_scale_cnn1d(cnn_input_1d, filters_1d, mode='add'):
    cnn1d = Conv1D(filters=filters_1d,kernel_size=1,strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_1d)
    cnn1d=BatchNormalization()(cnn1d)
    cnn1d_1=GELU(cnn1d)

    cnn1d = Conv1D(filters=filters_1d,kernel_size=3,strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_1d)
    cnn1d=BatchNormalization()(cnn1d)
    cnn1d_2=GELU(cnn1d)

    cnn1d = Conv1D(filters=filters_1d,kernel_size=5,strides=1,padding='same', kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input_1d)
    cnn1d=BatchNormalization()(cnn1d)
    cnn1d_3=GELU(cnn1d)

    if mode== 'add':
        cnn1d = add([cnn1d_1, cnn1d_2, cnn1d_3])
    else:
        cnn1d = concatenate([cnn1d_1, cnn1d_2, cnn1d_3],-1)
    return cnn1d


def CNN(frame,channels,batch_size,droprate,lr,activation):

    cnn_input = Input(shape=(frame, channels), name='the_inputs', dtype='float32')
    cnn_input_2d=Reshape((-1, channels, 1))(cnn_input)

    cnn2d= muti_scale_cnn2d(cnn_input_2d, filters_2d=16, strides=2, mode='add')
    channels = channels//2 if channels%2 == 0 else channels//2+1
    cnn2d= muti_scale_cnn2d(cnn2d, filters_2d=32, strides=2, mode='add')
    channels = channels//2 if channels%2 == 0 else channels//2+1
    cnn2d= muti_scale_cnn2d(cnn2d, filters_2d=64, strides=2, mode='add')
    channels = channels//2 if channels%2 == 0 else channels//2+1
    cnn2d=Reshape((-1, channels*64))(cnn2d)

    cnn2d = TimeDistributed(Dropout(0.2))(cnn2d)

    cnn=muti_scale_cnn1d(cnn2d, filters_1d=64, mode='add')


    cnn_pos = TrigPosEmbedding(output_dim=64,mode=TrigPosEmbedding.MODE_ADD)(cnn)

    cnn1 = resblock(cnn_pos, 64, 3, activation, 7, 4,batch_size)

    cnn_pos = TrigPosEmbedding(output_dim=64,mode=TrigPosEmbedding.MODE_ADD)(cnn1)
    cnn2 = resblock(cnn_pos, 64, 3, activation, 5, 4,batch_size)

    cnn_pos = TrigPosEmbedding(output_dim=64,mode=TrigPosEmbedding.MODE_ADD)(cnn2)

    cnn1 = resblock(cnn_pos, 64, 3, activation, 5, 4,batch_size)

    cnn_pos = TrigPosEmbedding(output_dim=64,mode=TrigPosEmbedding.MODE_ADD)(cnn1)
    cnn2 = resblock(cnn_pos, 64, 3, activation, 3, 4,batch_size)


    cnn3 = TimeDistributed(Dropout(0.2))(cnn2)

    cnn=muti_scale_cnn1d(cnn3, filters_1d=32, mode='concat')
    cnn = TimeDistributed(Dropout(0.5))(cnn)
    cnn_g=muti_scale_cnn1d(cnn, filters_1d=4, mode='concat')
    cnn = concatenate([cnn, cnn_g],-1)
    cnn_gg = GlobalAveragePooling1D()(cnn_g)

    cnn_gender=Dense(2,activation='softmax', kernel_initializer=TruncatedNormal(stddev=0.02), name='gender')(cnn_gg)

    cnn_output=TimeDistributed(Dense(6,activation='softmax', kernel_initializer=TruncatedNormal(stddev=0.02)), name='softmax')(cnn)
    print('\n',"USE:",'\n')

    model_optput = Model(inputs=[cnn_input],outputs=[cnn_output])
    model_optput.summary()
    model_optput_g = Model(inputs=[cnn_input],outputs=[cnn_output,cnn_gender])

    #ctc loss
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')
    ctc_loss = Lambda(ctc_loss_lambda, output_shape=(1,), name='ctc')([labels, cnn_output, input_length, label_length])

    model = Model(inputs=[cnn_input, labels, input_length, label_length],outputs=[ctc_loss,cnn_gender])

    test_func = K.function([cnn_input], [cnn_output])#0 = test, 1 = train

    return model, model_optput, test_func, model_optput_g

def tonet(train_data,train_label,test_data,test_label,test_label_f, test_label_n, test_label_s, frame,channels,droprate,lr,activation):

    val_content, val_label, val_input_length = get_data(data_path_file="test", root = '/data/gq/sjc/aishell-1/data_aishell/wav/test', stride=8, test_num=None)

    batch_size = 32

    model, model_optput, test_func, model_optput_g = CNN(frame,channels,batch_size,droprate,lr,activation)

    adabd=AdaBound(lr=lr,
                final_lr=0.1,
                gamma=1e-03,
                weight_decay=0.,
                amsbound=False)

    model.compile(optimizer=adabd,
        loss=[lambda y_true,y_pred: y_pred, 'categorical_crossentropy'],
        loss_weights=[0.9,0.1])

    print("training ==========~~~~~~~~=======")
    tensorBoard = TensorBoard(
                  log_dir='./log_dir',
                  batch_size=batch_size,
                  update_freq = 'epoch'
        )
    metric = MetricCallback(test_func, val_content, val_label, val_input_length, batch_size=1, test_num=None)
    #train data
    my_training_batch_generator = Generate_arrays_from_file("train",batch_size,stride=8,flag='train')
    #dev data
    my_validation_batch_generator = Generate_arrays_from_file("test",batch_size,stride=8,flag='test')
    # 保存full模型
    full_checkpointer = OutputModelCheckpoint(model, filepath="Models/full_tone.{epoch:04d}-{val_loss:.6f}.hdf5", verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')
    # 保存声调模型
    checkpointer = OutputModelCheckpoint(model_optput, filepath="Models/tone.{epoch:04d}-{val_loss:.6f}.hdf5", verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')
    checkpointer_g = OutputModelCheckpoint(model_optput_g, filepath="Models/tone_g.{epoch:04d}-{val_loss:.6f}.hdf5", verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')
    #model.fit(train_data, train_label, validation_split=0.1,shuffle=True, epochs=20,verbose=1,batch_size=128,callbacks=[checkpointer])  
    model.fit_generator(generator=my_training_batch_generator, steps_per_epoch=None, epochs=1000, verbose=1, 
                        callbacks=[full_checkpointer,checkpointer,checkpointer_g,metric,tensorBoard], validation_data=my_validation_batch_generator,
                         max_queue_size=1280, workers=16, use_multiprocessing=True,shuffle=True)
    K.get_session().graph.finalize()


def decode(y_pred, input_length):
    decoded = []
    prob = []
    for i in range(0,y_pred.shape[0]):

        decoded_batch = []
        prob_batch = []
        for j in range(0,y_pred.shape[1]):
            decoded_batch.append(np.argmax(y_pred[i][j]))
            prob_batch.append(y_pred[i][j][decoded_batch[-1]])
            if not input_length[i]:
                break
            input_length[i] -= 1

        temp = [k for k, g in groupby(decoded_batch)]
        temp[:] = [x for x in temp if x != [6]]
        decoded.append(np.array(temp, dtype='int32'))
        prob.append(np.array(prob_batch, dtype='float32').mean())
    return np.array(decoded), np.array(prob)

def decode_batch(test_func, batch, input_length,ctc_dc):

    output = test_func([batch])[0]

    tone, prob = ctc_dc.ctc_decode_tf(output, input_length)

    return tone[0], prob

def decode_batch_predict(test_func, batch, input_length,ctc_dc):


    output = test_func.predict([batch], batch_size=len(batch))

    tone, prob = ctc_dc.ctc_decode_tf(output, input_length)

    return tone[0], prob


class Ctc_Decode:
    # 用tf定义一个专门ctc解码的图和会话，就不会一直增加节点了，速度快了很多
    def __init__(self ,batch_size, timestep, nclass):
        self.batch_size = batch_size
        self.timestep = timestep
        self.nclass = nclass
        self.graph_ctc = tf.Graph()
        with self.graph_ctc.as_default():
             with tf.device('/cpu:0'):
                self.y_pred_tensor = tf.placeholder(shape=[self.batch_size, self.timestep, self.nclass], dtype=tf.float32, name="y_pred_tensor")
                self.input_length_tensor = tf.placeholder(shape=[self.batch_size,], dtype=tf.int32, name="input_length_tensor")

                self.decoded_sequences, self.prob = self.ctc_decode(self.y_pred_tensor, self.input_length_tensor, greedy=True, beam_width=100,
                                                        top_paths=1, merge_repeated=True)

                self.ctc_sess = tf.Session(graph=self.graph_ctc)

    def ctc_decode_tf(self, y_pred, input_length):
        #y_pred, input_length = args
        decoded_sequences, prob = self.ctc_sess.run([self.decoded_sequences,self.prob],
                                     feed_dict={self.y_pred_tensor: y_pred, self.input_length_tensor: input_length})
        return decoded_sequences, prob

    def ctc_decode(self, y_pred, input_length, greedy=True, beam_width=100,
               top_paths=1, merge_repeated=False):
        
        """Decodes the output of a softmax.
        Can use either greedy search (also known as best path)
        or a constrained dictionary search.
        # Arguments
            y_pred: tensor `(samples, time_steps, num_categories)`
                containing the prediction, or output of the softmax.
            input_length: tensor `(samples, )` containing the sequence length for
                each batch item in `y_pred`.
            greedy: perform much faster best-path search if `True`.
                This does not use a dictionary.
            beam_width: if `greedy` is `False`: a beam search decoder will be used
                with a beam of this width.
            top_paths: if `greedy` is `False`,
                how many of the most probable paths will be returned.
            merge_repeated: if `greedy` is `False`,
                merge repeated classes in the output beams.
        # Returns
            Tuple:
                List: if `greedy` is `True`, returns a list of one element that
                    contains the decoded sequence.
                    If `False`, returns the `top_paths` most probable
                    decoded sequences.
                    Important: blank labels are returned as `-1`.
                Tensor `(top_paths, )` that contains
                    the log probability of each decoded sequence.
        """
        _EPSILON = 1e-7
        y_pred = tf_math_ops.log(tf.transpose(y_pred, perm=[1, 0, 2]) + _EPSILON)
        input_length = tf.cast(input_length, tf.int32)

        if greedy:
            (decoded, log_prob) = ctc_ops.ctc_greedy_decoder(
                inputs=y_pred,
                sequence_length=input_length)
        else:
            (decoded, log_prob) = ctc_ops.ctc_beam_search_decoder(
                inputs=y_pred,
                sequence_length=input_length, beam_width=beam_width,
                top_paths=top_paths, merge_repeated=merge_repeated)

        decoded_dense = []
        for st in decoded:
            dense_tensor = tf.sparse.to_dense(st, default_value=-1)
            decoded_dense.append(dense_tensor)
        return decoded_dense, log_prob

#ctc 解码
def decode_ctc(y_pred, input_length):
    input_length = K.variable(input_length)
    ret = K.ctc_decode(y_pred, input_length, greedy = True, beam_width=100, top_paths=10)
    ret = K.get_value(ret)
    print(ret,2)
    tone = ret[0][0]
    prob = ret[0][1]
    return tone, prob

def predict(model_path, file_path, frame,channels,droprate,lr,activation):
    from keras.models import load_model
    from keras.utils import np_utils
    import time
    batch_size = 1
    _, model, test_func, model_g = CNN(frame,channels,batch_size,droprate,lr,activation)
    model.load_weights(model_path)

    test_data,test_label,input_length=get_data(file_path, root = '/data/gq/sjc/aishell-1/data_aishell/wav/test', stride=8, test_num=None)

    ctc_class = Ctc_Decode(batch_size=None,timestep=None, nclass=6)

    print('this is a test \n')
    test_num = len(test_data)
    y_pred = []
    prob = []

    s = time.time()
    for i in range(test_num):
        inputs = np.array([test_data[i]])
        y_pred_tmp, prob_tmp = decode_batch_predict(model, inputs,[input_length[i]],ctc_class)
        y_pred.extend(y_pred_tmp)
        prob.extend(prob_tmp)
    e = time.time()
    print('time:',e-s)

    y_pred = [''.join([str(int(p)).replace('-1', '') for p in pred]) for pred in y_pred]
    y_true = [''.join([str(t) for t in true]) for true in test_label[0:test_num]]
    with open('tmp6', 'a', encoding='utf-8') as f:
        f.write( str(list(zip(y_pred, y_true))) + '\n' )
    _, mean, ops, t_each_tone = wers2(y_true, y_pred)
    print('WER(%s):'%(test_num), mean,'\n')
    print('ops:',ops)
    print('each_tone:',t_each_tone)
    del y_pred_tmp
    del prob_tmp


def main():
    #train_data,train_label=get_data('train')
    train_data,train_label,test_data,test_label,test_label_f, test_label_n, test_label_s=[],[],[],[],[],[],[]
    # test_data,test_label,test_label_f, test_label_n, test_label_s=get_data('test28')
    frame = None#470
    channels = 43
    droprate = 0.1
    lr = 0.001
    activation = 'relu'
    #tonet(train_data,train_label,test_data,test_label,test_label_f, test_label_n, test_label_s, frame,channels,droprate,lr,activation)

    predict("Models/tone.0193-4.304704.hdf5", 'test', frame,channels,droprate,lr,activation)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
main()

