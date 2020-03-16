import keys
#import test_key as keys
import torch
import numpy as np
import denseNet_TF2
import denseNet_pytorch
import tensorflow as tf
from collections import OrderedDict
from tensorflow.python.framework import graph_util

def state_dict_layer_names(state_dict):
      layer_names = [".".join(k.split('.')[:-1]) for k in state_dict.keys()]
      return list(OrderedDict.fromkeys(layer_names))

def check_for_missing_layers(tf_layer_lists, pytorch_layer_lists):
    tf_layers = []
    for layer in tf_layer_lists:
        name = layer.name.split('/')[0]
        if name not in tf_layers:
            tf_layers.append(name)

    if not all(x in tf_layers for x in pytorch_layer_lists):
        missing_layers = list(set(pytorch_layer_lists) - set(tf_layers))
        raise Exception("Missing layer(s) in Keras HDF5 that are present" +
                        " in state_dict: {}".format(missing_layers))

def pytorch_to_tf(pytorch_model, tf_model, tf_layer_lists):
    tf_var_map = {}
    tf_var_list = []
    for layer in tf_layer_lists:
        tf_var_map[layer.name] = layer

    pytorch_layer_dicts = pytorch_model.state_dict()
    pytorch_layer_lists = state_dict_layer_names(pytorch_layer_dicts)
    check_for_missing_layers(tf_layer_lists, pytorch_layer_lists)

    for layer in pytorch_layer_lists:
        weight_key = layer + '.weight'
        bias_key = layer + '.bias'
        running_mean_key = layer + '.running_mean'
        running_var_key = layer + '.running_var'
        lstm_weight_ih = layer + '.weight_ih_l0'
        lstm_weight_hh = layer + '.weight_hh_l0'
        lstm_bias_ih = layer + '.bias_ih_l0'
        lstm_bias_hh = layer + '.bias_hh_l0'
        lstm_weight_ih_reverse = layer + '.weight_ih_l0_reverse'
        lstm_weight_hh_reverse = layer + '.weight_hh_l0_reverse'
        lstm_bias_ih_reverse = layer + '.bias_ih_l0_reverse'
        lstm_bias_hh_reverse = layer + '.bias_hh_l0_reverse'
        if weight_key in pytorch_layer_dicts:
            weights = pytorch_layer_dicts[weight_key].numpy()
            weights = convert_weights(weights)

            if layer.find("bn") == -1:
                weights_tf = layer + '/weights:0'
            else:
                weights_tf = layer + '/gamma:0'
            var = tf_var_map[weights_tf].assign(tf.constant(weights))
            tf_var_list.append(var)

        if bias_key in pytorch_layer_dicts:
            bias = pytorch_layer_dicts[bias_key].numpy()
            if layer.find("bn") == -1:
                bias_tf = layer + '/bias:0'
            else:
                bias_tf = layer + '/beta:0'
            var = tf_var_map[bias_tf].assign(tf.constant(bias))
            tf_var_list.append(var)

        if running_var_key in pytorch_layer_dicts:
            running_var = pytorch_layer_dicts[running_var_key].numpy()
            moving_variance_tf = layer + '/moving_variance:0'
            var = tf_var_map[moving_variance_tf].assign(tf.constant(running_var))
            tf_var_list.append(var)

        if running_mean_key in pytorch_layer_dicts:
            running_mean = pytorch_layer_dicts[running_mean_key].numpy()
            moving_mean_tf = layer + '/moving_mean:0'
            var = tf_var_map[moving_mean_tf].assign(tf.constant(running_mean))
            tf_var_list.append(var)

        if lstm_weight_ih in pytorch_layer_dicts:
            weights = pytorch_layer_dicts[lstm_weight_ih].numpy()
            weights = convert_weights(weights)
            weights_tf = layer + '/fw/kernel-i:0'
            var = tf_var_map[weights_tf].assign(tf.constant(weights))
            tf_var_list.append(var)

        if lstm_weight_ih_reverse in pytorch_layer_dicts:
            weights = pytorch_layer_dicts[lstm_weight_ih_reverse].numpy()
            weights = convert_weights(weights)
            weights_tf = layer + '/bw/kernel-i:0'
            var = tf_var_map[weights_tf].assign(tf.constant(weights))
            tf_var_list.append(var)

        if lstm_weight_hh in pytorch_layer_dicts:
            weights = pytorch_layer_dicts[lstm_weight_hh].numpy()
            weights = convert_weights(weights)
            weights_tf = layer + '/fw/kernel-h:0'
            var = tf_var_map[weights_tf].assign(tf.constant(weights))
            tf_var_list.append(var)

        if lstm_weight_hh_reverse in pytorch_layer_dicts:
            weights = pytorch_layer_dicts[lstm_weight_hh_reverse].numpy()
            weights = convert_weights(weights)
            weights_tf = layer + '/bw/kernel-h:0'
            var = tf_var_map[weights_tf].assign(tf.constant(weights))
            tf_var_list.append(var)

        if lstm_bias_ih in pytorch_layer_dicts:
            bias = pytorch_layer_dicts[lstm_bias_ih].numpy()
            bias_tf = layer + '/fw/bias-i:0'
            var = tf_var_map[bias_tf].assign(tf.constant(bias))
            tf_var_list.append(var)

        if lstm_bias_ih_reverse in pytorch_layer_dicts:
            bias = pytorch_layer_dicts[lstm_bias_ih_reverse].numpy()
            bias_tf = layer + '/bw/bias-i:0'
            var = tf_var_map[bias_tf].assign(tf.constant(bias))
            tf_var_list.append(var)

        if lstm_bias_hh in pytorch_layer_dicts:
            bias = pytorch_layer_dicts[lstm_bias_hh].numpy()
            bias_tf = layer + '/fw/bias-h:0'
            var = tf_var_map[bias_tf].assign(tf.constant(bias))
            tf_var_list.append(var)
       
        if lstm_bias_hh_reverse in pytorch_layer_dicts:
            bias = pytorch_layer_dicts[lstm_bias_hh_reverse].numpy()
            bias_tf = layer + '/bw/bias-h:0'
            var = tf_var_map[bias_tf].assign(tf.constant(bias))
            tf_var_list.append(var)
    
    return tf_var_list

def convert_weights(weights):
    if len(weights.shape) == 4: 
        weights = weights.transpose(2, 3, 1, 0)
    if len(weights.shape) == 2:
        weights = weights.transpose()

    return weights

alphabet = keys.alphabet.decode('utf-8')
nclass = len(alphabet) + 1
model_pytorch = denseNet_pytorch.DenseNet3(32, nclass, 128)

model_path = './CRNN.pth'
#model_path = './netCRNN_9.pth'
state_dict = torch.load(model_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
    #print name,v.shape,v
model_pytorch.load_state_dict(new_state_dict)

img = tf.placeholder(tf.float32, shape=[1, None, None,3])
model_tf = denseNet_TF2.DenseNet()
pred = model_tf.build_model(img)
tf_layer_lists = [layer for layer in tf.global_variables()]
tf_layers = pytorch_to_tf(model_pytorch, model_tf, tf_layer_lists)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for layer in tf_layers:
        sess.run(layer)
    graph = tf.get_default_graph().as_graph_def() 
    #node_names = [node.name for node in graph.node]
    #f = file('Node.txt', "w")
    #for x in node_names:
    #    f.write(x)
    #    f.write('\n')
    constant_graph = graph_util.convert_variables_to_constants(sess, graph, ['add_1'])
    with tf.gfile.GFile('model.pb', 'wb') as f:
        f.write(constant_graph.SerializeToString())

