from time import time

import numpy as np
import torch
from torchsummary import summary

from plain_unet import UNet
from nested_unet import UNet_new, NestedUNet
from cgnet import Context_Guided_Network
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from custom_unet import FastNestedUNet

if __name__ == '__main__':
    # hyperparameters
    input_shape = (512, 512)

    # compile model
    model1 = UNet()  # 31,037,633 params
    model2 = UNet_new(1)  # 7,852,545 params
    model3 = NestedUNet(1)  # 9,163,329 params
    model4 = Context_Guided_Network(1)  # 491,698 params
    model5 = MobileNetV3_Small(1)
    model6 = MobileNetV3_Large(1)
    model7 = FastNestedUNet()

    # run torch summary report
    # summary(model1, input_size=(3, input_shape[0], input_shape[1]))
    # summary(model2, input_size=(3, input_shape[0], input_shape[1]))
    # summary(model3, input_size=(3, input_shape[0], input_shape[1]))
    # summary(model4, input_size=(3, input_shape[0], input_shape[1]))

    # ------- TESTING MODEL INFERENCE TIMES -------- #

    image = np.zeros((8, 3, 512, 512), dtype=np.float32)
    image = torch.tensor(image).cpu()

    start = time()
    output = model2(image)
    print('UNet: {}'.format(time() - start))

    start = time()
    output = model3(image)
    print('UNet++: {}'.format(time() - start))

    start = time()
    output = model4(image)
    print('ContextGuidedNetwork: {}'.format(time() - start))

    start = time()
    output = model5(image)
    print('MobileNetV3 Small: {}'.format(time() - start))

    start = time()
    output = model6(image)
    print('MobileNetV3 Large: {}'.format(time() - start))

    start = time()
    output = model7(image)
    print('FastNestedUNet: {}'.format(time() - start))


