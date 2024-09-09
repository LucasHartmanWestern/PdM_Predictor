from torchsummary import summary

from plain_unet import UNet
from nested_unet import UNet_new, NestedUNet
from cgnet import Context_Guided_Network


if __name__ == '__main__':
    # hyperparameters
    input_shape = (512, 512)

    # compile model
    model1 = UNet()  # 31,037,633 params
    model2 = UNet_new(1)  # 7,852,545 params
    model3 = NestedUNet(1)  # 9,163,329 params
    model4 = Context_Guided_Network(1)  # 491,698 params

    # run torch summary report
    # summary(model1, input_size=(3, input_shape[0], input_shape[1]))
    # summary(model2, input_size=(3, input_shape[0], input_shape[1]))
    # summary(model3, input_size=(3, input_shape[0], input_shape[1]))
    summary(model4, input_size=(3, input_shape[0], input_shape[1]))


