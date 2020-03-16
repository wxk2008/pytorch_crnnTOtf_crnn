import cv2
import keys

import torch
import utils
import dataset
import denseNet
from PIL import Image
from torch.autograd import Variable
from collections import OrderedDict

model_path = './CRNN.pth'
img_path = './data/test-14.jpg'
alphabet = keys.alphabet.decode('utf-8')

nclass = len(alphabet) + 1
model  = denseNet.DenseNet3(32, nclass, 128)

state_dict = torch.load(model_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)

converter = utils.strLabelConverter(alphabet)

img = cv2.imread(img_path)
W = img.shape[1]*(32.0/img.shape[0])

transformer = dataset.resizeNormalize((int(W), 32))
image = Image.open(img_path)
image = transformer(image)
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.squeeze(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
