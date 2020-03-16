import os
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BasicLSTMCell():
    def __init__(self, hidden, input,weights_i,biases_i,weights_h,biases_h):
        self.hidden = hidden
        self.input = input
        self.weights_i = weights_i
        self.biases_i = biases_i
        self.weights_h = weights_h
        self.biases_h = biases_h

        self.init_h = tf.matmul(self.input[0, :, :], tf.zeros([self.input.shape[2], self.hidden]))
        self.init_c = self.init_h
        self.previous_state = tf.stack([self.init_h, self.init_c])

    def Step(self, previous_hc, current_input):
        h, c = tf.unstack(previous_hc)
        res_i = tf.matmul(current_input, self.weights_i)
        w_i = tf.nn.bias_add(res_i, self.biases_i)
        res_h = tf.matmul(h, self.weights_h)
        w_h = tf.nn.bias_add(res_h, self.biases_h)
        concat = w_i + w_h 
        i, f, j, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
        new_c = (c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        new_state = tf.stack([new_h, new_c])
        return new_state

def BLSTM(cell_fw, cell_bw):
    h_f = tf.scan(fn=cell_fw.Step, elems=cell_fw.input, initializer=cell_fw.previous_state, name='fw')[:, 0, :, :]
    h_b = tf.scan(fn=cell_bw.Step, elems=tf.reverse(cell_bw.input, axis=[0]), initializer=cell_bw.previous_state, name='bw')[:, 0, :, :]
    h_b = tf.reverse(h_b, axis=[0])
    b_lstm = tf.concat(values=[h_f, h_b], axis=2)
    return b_lstm

class DenseNet():
    def __init__(self, growth_rate=12,reduction=0.5):
        self.growth_rate = growth_rate
        self.reduction = reduction
    
    def BottleneckBlock(self, input_img, in_planes, scope):
        inter_planes = self.growth_rate * 4
        with tf.name_scope(scope):
            x = slim.batch_norm(input_img, scale=True, is_training=False, scope=scope+'.bn1')
            x = tf.nn.relu(x)
            x = slim.conv2d(x, inter_planes, kernel_size=(1,1), stride=1, padding='VALID',
                            activation_fn=None, biases_initializer=None, trainable=False, scope=scope+'.conv1') 
            x = slim.batch_norm(x, scale=True, is_training=False, scope=scope+'.bn2')
            x = tf.nn.relu(x)
            x = slim.conv2d(x, self.growth_rate, kernel_size=(3,3), stride=1, padding='SAME',
                            activation_fn=None, biases_initializer=None, trainable=False, scope=scope+'.conv2') 
            return x 
    def TransitionBlock(self, input_img, out_planes, scope):
        with tf.name_scope(scope):
            x = slim.batch_norm(input_img, scale=True, is_training=False, scope=scope+'.bn1')
            x = tf.nn.relu(x)
            x = slim.conv2d(x, out_planes, kernel_size=(1,1), stride=1, padding='VALID',
                            activation_fn=None, biases_initializer=None, trainable=False, scope=scope+'.conv1') 
            x = slim.avg_pool2d(x, 2)
            return x 
    
    def DenseBlock(self, input_img, in_planes, scope):
        with tf.name_scope(scope):
            concat = []
            concat.append(input_img)

            x = self.BottleneckBlock(input_img, in_planes+0*self.growth_rate, scope+'.layer.0')
            concat.append(x)

            x = tf.concat(concat,axis=3)
            x = self.BottleneckBlock(x, in_planes+1*self.growth_rate, scope+'.layer.1')
            concat.append(x)

            x = tf.concat(concat,axis=3)
            x = self.BottleneckBlock(x, in_planes+2*self.growth_rate, scope+'.layer.2')
            concat.append(x)

            x = tf.concat(concat,axis=3)
            return x
        
    def build_model(self, input_img):
        in_planes = 2 * self.growth_rate
        net = slim.conv2d(input_img, in_planes, kernel_size=(3,3), stride=1, padding='SAME',
                          activation_fn=None, biases_initializer=None, trainable=False, scope='conv1')  
   
        net = self.DenseBlock(net,in_planes,'block1')
        in_planes = int(in_planes+3*self.growth_rate)
        net = self.TransitionBlock(net,int(math.floor(in_planes*self.reduction)),'trans1') 
        in_planes = int(math.floor(in_planes*self.reduction))

        net = self.DenseBlock(net,in_planes,'block2') 
        in_planes = int(in_planes+3*self.growth_rate)
        net = self.TransitionBlock(net,int(math.floor(in_planes*self.reduction)),'trans2') 
        in_planes = int(math.floor(in_planes*self.reduction))

        net = self.DenseBlock(net,in_planes,'block3') 
       
        net = slim.batch_norm(net,scale=True, is_training=False, scope='bn1')
        net = tf.nn.relu(net) 
        net = slim.avg_pool2d(net, [8,1], stride=1)          
        net = tf.squeeze(net, [1])
        net = tf.transpose(net, perm=[1,0,2])

        weights_i = tf.get_variable("rnn.0.rnn/fw/kernel-i",shape=[net.get_shape()[2].value,4*128],dtype=tf.float32,trainable=False)
        biases_i = tf.get_variable("rnn.0.rnn/fw/bias-i",shape=[4*128],dtype=tf.float32,trainable=False)
        weights_h = tf.get_variable("rnn.0.rnn/fw/kernel-h",shape=[128,4*128],dtype=tf.float32,trainable=False)
        biases_h = tf.get_variable("rnn.0.rnn/fw/bias-h",shape=[4*128],dtype=tf.float32,trainable=False)
        lstm_fw_1 = BasicLSTMCell(128,net,weights_i,biases_i,weights_h,biases_h)

        weights_i = tf.get_variable("rnn.0.rnn/bw/kernel-i",shape=[net.get_shape()[2].value,4*128],dtype=tf.float32,trainable=False)
        biases_i = tf.get_variable("rnn.0.rnn/bw/bias-i",shape=[4*128],dtype=tf.float32,trainable=False)
        weights_h = tf.get_variable("rnn.0.rnn/bw/kernel-h",shape=[128,4*128],dtype=tf.float32,trainable=False)
        biases_h = tf.get_variable("rnn.0.rnn/bw/bias-h",shape=[4*128],dtype=tf.float32,trainable=False)
        lstm_bw_1 = BasicLSTMCell(128,net,weights_i,biases_i,weights_h,biases_h)
        blstm_1 = BLSTM(lstm_fw_1, lstm_bw_1)
        blstm_1 = tf.squeeze(blstm_1, [1])

        #blstm_concat1 = tf.squeeze(blstm_concat1, [1])
        w1 = tf.get_variable( "rnn.0.embedding/weights",shape=[256,128],dtype=tf.float32,trainable=False)
        b1 = tf.get_variable( "rnn.0.embedding/bias",shape=[128],dtype=tf.float32,trainable=False)
        net = tf.matmul(blstm_1, w1) + b1
        net = tf.expand_dims(net,1)

        weights_i = tf.get_variable("rnn.1.rnn/fw/kernel-i",shape=[128,4*128],dtype=tf.float32,trainable=False)
        biases_i = tf.get_variable("rnn.1.rnn/fw/bias-i",shape=[4*128],dtype=tf.float32,trainable=False)
        weights_h = tf.get_variable("rnn.1.rnn/fw/kernel-h",shape=[128,4*128],dtype=tf.float32,trainable=False)
        biases_h = tf.get_variable("rnn.1.rnn/fw/bias-h",shape=[4*128],dtype=tf.float32,trainable=False)
        lstm_fw_2 = BasicLSTMCell(128,net,weights_i,biases_i,weights_h,biases_h)

        weights_i = tf.get_variable("rnn.1.rnn/bw/kernel-i",shape=[128,4*128],dtype=tf.float32,trainable=False)
        biases_i = tf.get_variable("rnn.1.rnn/bw/bias-i",shape=[4*128],dtype=tf.float32,trainable=False)
        weights_h = tf.get_variable("rnn.1.rnn/bw/kernel-h",shape=[128,4*128],dtype=tf.float32,trainable=False)
        biases_h = tf.get_variable("rnn.1.rnn/bw/bias-h",shape=[4*128],dtype=tf.float32,trainable=False)
        lstm_bw_2 = BasicLSTMCell(128,net,weights_i,biases_i,weights_h,biases_h)
        blstm_2 = BLSTM(lstm_fw_2, lstm_bw_2)
        blstm_2 = tf.squeeze(blstm_2, [1])
        
        w2 = tf.get_variable( "rnn.1.embedding/weights",shape=[256,7509],dtype=tf.float32,trainable=False)
        b2 = tf.get_variable( "rnn.1.embedding/bias",shape=[7509],dtype=tf.float32,trainable=False)
        net = tf.matmul(blstm_2, w2) + b2
        #net = tf.expand_dims(net,1)

        return net

#sess = tf.Session()
#img = tf.placeholder(tf.float32, shape=[1,32,346,3])
#model = DenseNet()
#pred = model.build_model(img)
#sess.run(tf.global_variables_initializer())
#var = [v for v in tf.global_variables()]
#for v in var:
#   print(v.name,v.shape)
