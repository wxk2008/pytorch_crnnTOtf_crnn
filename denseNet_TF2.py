import os
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=0.0, state_is_tuple=True, activation=None, reuse=None):
        super(BasicLSTMCell,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return 2*self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self,input_img,state):
        c, h = tf.split( value=state, num_or_size_splits=2, axis=1)
        weights_i = tf.get_variable( "kernel-i",shape=[input_img.get_shape()[1].value,4*self._num_units],
                                     dtype=tf.float32,initializer=None,trainable=False)
        res_i = tf.matmul(input_img,weights_i)
        biases_i = tf.get_variable("bias-i",shape=[4*self._num_units],dtype=tf.float32, 
                                    initializer=None,trainable=False)
        w_i = tf.nn.bias_add(res_i, biases_i)


        weights_h = tf.get_variable( "kernel-h",shape=[h.get_shape()[1].value,4*self._num_units],
                                    dtype=tf.float32,initializer=None,trainable=False)
        res_h = tf.matmul(h,weights_h)
        biases_h = tf.get_variable("bias-h",shape=[4*self._num_units],dtype=tf.float32,
                                    initializer=None,trainable=False)
        w_h = tf.nn.bias_add(res_h, biases_h)
        concat = w_i + w_h

        i, f, j, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = ( c*tf.sigmoid(f+self._forget_bias) + tf.sigmoid(i)*self._activation(j))
        new_h = self._activation(new_c) * tf.sigmoid(o)

        new_state = tf.concat(values=[new_c,new_h],axis=1 )
        return new_h, new_state

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
	#net = tf.squeeze(net, [1])


        lstm_fw1 = BasicLSTMCell(128)
        lstm_bw1 = BasicLSTMCell(128)
        blstm1, bstate1 = tf.nn.bidirectional_dynamic_rnn(lstm_fw1, lstm_bw1, net, dtype=tf.float32, time_major=True, scope='rnn.0.rnn')
        blstm_concat1 = tf.concat(blstm1, 2)

        blstm_concat1 = tf.squeeze(blstm_concat1, [1])
        w1 = tf.get_variable( "rnn.0.embedding/weights",shape=[256,128],dtype=tf.float32,initializer=None)
        b1 = tf.get_variable( "rnn.0.embedding/bias",shape=[128],dtype=tf.float32,initializer=None)
        net = tf.matmul(blstm_concat1, w1) + b1

        net = tf.expand_dims(net,1)
        lstm_fw2 = BasicLSTMCell(128)
        lstm_bw2 = BasicLSTMCell(128)
        blstm2, bstate2 = tf.nn.bidirectional_dynamic_rnn(lstm_fw2, lstm_bw2, net, dtype=tf.float32, time_major=True, scope='rnn.1.rnn')
        blstm_concat2 = tf.concat(blstm2, 2)
        blstm_concat2 = tf.squeeze(blstm_concat2, [1])
        w2 = tf.get_variable( "rnn.1.embedding/weights",shape=[256,7509],dtype=tf.float32,initializer=None)
        b2 = tf.get_variable( "rnn.1.embedding/bias",shape=[7509],dtype=tf.float32,initializer=None)
        net = tf.matmul(blstm_concat2, w2) + b2
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
