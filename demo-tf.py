import os
import cv2
import keys
#import test_key as keys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-' 

    def decode(self, t, length):
        char_list = []
        for i in range(length):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                char_list.append(self.alphabet[t[i] - 1])
        return ''.join(char_list)

alphabet = keys.alphabet.decode('utf-8')
converter = strLabelConverter(alphabet)

img = cv2.imread('./data/054._11.jpg')
W = img.shape[1]*(32.0/img.shape[0])
img = cv2.resize(img,(int(W), 32),interpolation=cv2.INTER_LINEAR)
img = 2*(img/255.0 - 0.5)
img = np.array([img])

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
with gfile.FastGFile('./model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
sess.run(tf.global_variables_initializer())

input_img = sess.graph.get_tensor_by_name('Placeholder:0')
output = sess.graph.get_tensor_by_name('add_1:0')
output = sess.run([output],feed_dict = {input_img:img})

preds = tf.argmax(output[0], axis=1)
preds_1 = sess.run(preds)
sim_pred = converter.decode(preds_1, len(preds_1))
print sim_pred
