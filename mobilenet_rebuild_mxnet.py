from mxnet import nd
import mxnet as mx
import cv2
import numpy as np
import tvm
import math

ctx = mx.cpu()
shape=300
image_file='/home/prdcv/PycharmProjects/gvh205/others/images/dog.jpg'
weight_dir='/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/fp32_origin/'
bias_dir='/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/fp32_origin/'

def load_image(image_file):
    origimg = cv2.imread(image_file)
    img = cv2.resize(origimg, (shape, shape))
    img = np.array(img) - np.array([123., 117., 104.])
    img = img * 0.007843

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    x = img[np.newaxis, :]
    return x

def get_weight(layer_name, weight_shape):
    weight = np.fromfile(weight_dir+layer_name+'_weight.bin', dtype='float32')
    return np.reshape(weight,weight_shape)

def get_bias(layer_name, bias_shape):
    return np.fromfile(bias_dir+layer_name+'_bias.bin', dtype='float32')


def convolution(input, w, b, kernel, stride, pad=(1,1), dw=False, relu=True):
    if(dw==True):
        group=b.shape[0]
    else:
        group=1
    conv=nd.Convolution(data=nd.array(input), weight=nd.array(w), bias=nd.array(b), kernel=kernel, pad=pad, stride=stride, num_filter=b.shape[0], num_group=group)
    if(relu==False):
        return conv.asnumpy()
    else:
        return nd.relu(conv).asnumpy()


def get_mbox_conf(mbox_1,mbox_2,mbox_3,mbox_4,mbox_5,mbox_6):
    transpose1=mbox_1.transpose(0,2,3,1)
    flatten1=transpose1.reshape(1,mbox_1.shape[1]*mbox_1.shape[2]*mbox_1.shape[3]*mbox_1.shape[0])
    transpose2=mbox_2.transpose(0,2,3,1)
    flatten2=transpose2.reshape(1,mbox_2.shape[1]*mbox_2.shape[2]*mbox_2.shape[3]*mbox_2.shape[0])
    transpose3=mbox_3.transpose(0,2,3,1)
    flatten3=transpose3.reshape(1,mbox_3.shape[1]*mbox_3.shape[2]*mbox_3.shape[3]*mbox_3.shape[0])
    transpose4=mbox_4.transpose(0,2,3,1)
    flatten4=transpose4.reshape(1,mbox_4.shape[1]*mbox_4.shape[2]*mbox_4.shape[3]*mbox_4.shape[0])
    transpose5=mbox_5.transpose(0,2,3,1)
    flatten5=transpose5.reshape(1,mbox_5.shape[1]*mbox_5.shape[2]*mbox_5.shape[3]*mbox_5.shape[0])
    transpose6=mbox_6.transpose(0,2,3,1)
    flatten6=transpose6.reshape(1,mbox_6.shape[1]*mbox_6.shape[2]*mbox_6.shape[3]*mbox_6.shape[0])
    concat=np.concatenate((flatten1,flatten2,flatten3,flatten4,flatten5,flatten6),axis=1)
    return concat

def get_mbox_loc(mbox_1,mbox_2,mbox_3,mbox_4,mbox_5,mbox_6):
    transpose1=mbox_1.transpose(0,2,3,1)
    flatten1=transpose1.reshape(1,mbox_1.shape[1]*mbox_1.shape[2]*mbox_1.shape[3]*mbox_1.shape[0])
    transpose2=mbox_2.transpose(0,2,3,1)
    flatten2=transpose2.reshape(1,mbox_2.shape[1]*mbox_2.shape[2]*mbox_2.shape[3]*mbox_2.shape[0])
    transpose3=mbox_3.transpose(0,2,3,1)
    flatten3=transpose3.reshape(1,mbox_3.shape[1]*mbox_3.shape[2]*mbox_3.shape[3]*mbox_3.shape[0])
    transpose4=mbox_4.transpose(0,2,3,1)
    flatten4=transpose4.reshape(1,mbox_4.shape[1]*mbox_4.shape[2]*mbox_4.shape[3]*mbox_4.shape[0])
    transpose5=mbox_5.transpose(0,2,3,1)
    flatten5=transpose5.reshape(1,mbox_5.shape[1]*mbox_5.shape[2]*mbox_5.shape[3]*mbox_5.shape[0])
    transpose6=mbox_6.transpose(0,2,3,1)
    flatten6=transpose6.reshape(1,mbox_6.shape[1]*mbox_6.shape[2]*mbox_6.shape[3]*mbox_6.shape[0])
    concat=np.concatenate((flatten1,flatten2,flatten3,flatten4,flatten5,flatten6),axis=1)
    return concat

def get_mbox_prior(priorBox_1,priorBox_2,priorBox_3,priorBox_4,priorBox_5,priorBox_6):

    output_priorBox_1 = get_prior_output(priorBox_1, sizes=(0.2,), ratios=(1.0, 2.0, 0.5), steps=(-0.003333, -0.003333))
    output_priorBox_2 = get_prior_output(priorBox_2, sizes=(0.35, 0.41833), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_3 = get_prior_output(priorBox_3, sizes=(0.5, 0.570088), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_4 = get_prior_output(priorBox_4, sizes=(0.65, 0.72111), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_5 = get_prior_output(priorBox_5, sizes=(0.8, 0.87178), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_6 = get_prior_output(priorBox_6, sizes=(0.95, 0.974679), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))

    flatten1=output_priorBox_1.reshape(1,output_priorBox_1.shape[1]*output_priorBox_1.shape[2])
    flatten2=output_priorBox_2.reshape(1,output_priorBox_2.shape[1]*output_priorBox_2.shape[2])
    flatten3=output_priorBox_3.reshape(1,output_priorBox_3.shape[1]*output_priorBox_3.shape[2])
    flatten4=output_priorBox_4.reshape(1,output_priorBox_4.shape[1]*output_priorBox_4.shape[2])
    flatten5=output_priorBox_5.reshape(1,output_priorBox_5.shape[1]*output_priorBox_5.shape[2])
    flatten6=output_priorBox_6.reshape(1,output_priorBox_6.shape[1]*output_priorBox_6.shape[2])
    concat=np.concatenate((flatten1,flatten2,flatten3,flatten4,flatten5,flatten6),axis=1)
    multibox_prior=concat.reshape(1,1917,4)
    return multibox_prior

def get_prior_output(input_data, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    dshape=input_data.shape
    dtype = 'float32'

    in_height = dshape[2]
    in_width = dshape[3]
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    size_ratio_concat = sizes + ratios
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
    np_out = np.zeros(oshape).astype(dtype)

    for i in range(in_height):
        center_h = (i + offset_h) * steps_h
        for j in range(in_width):
            center_w = (j + offset_w) * steps_w
            for k in range(num_sizes + num_ratios - 1):
                w = size_ratio_concat[k] * in_height / in_width / 2.0 if k < num_sizes else \
                    size_ratio_concat[0] * in_height / in_width * math.sqrt(size_ratio_concat[k + 1]) / 2.0
                h = size_ratio_concat[k] / 2.0 if k < num_sizes else \
                    size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                count = i * in_width * (num_sizes + num_ratios - 1) + j * (num_sizes + num_ratios - 1) + k
                np_out[0][count][0] = center_w - w
                np_out[0][count][1] = center_h - h
                np_out[0][count][2] = center_w + w
                np_out[0][count][3] = center_h + h
    if clip:
        np_out = np.clip(np_out, 0, 1)
    return np_out










data        =load_image(image_file)

conv0       =convolution(input=data, w=get_weight('conv0',(32,3,3,3)),b=get_bias('conv0',(32,)), kernel=(3,3), stride=(2, 2))
conv1_dw    =convolution(input=conv0, w=get_weight('conv1_dw',(32,1,3,3)), b=get_bias('conv1_dw',(32,)), kernel=(3,3), stride=(1, 1), dw=True)
conv1       =convolution(input=conv1_dw, w=get_weight('conv1',(64,32,1,1)), b=get_bias('conv1',(64,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv2_dw    =convolution(input=conv1, w=get_weight('conv2_dw',(64,1,3,3)), b=get_bias('conv2_dw',(64,)), kernel=(3,3), stride=(2, 2), dw=True)
conv2       =convolution(input=conv2_dw, w=get_weight('conv2',(128,64,1,1)), b=get_bias('conv2',(128,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv3_dw    =convolution(input=conv2, w=get_weight('conv3_dw',(128,1,3,3)), b=get_bias('conv3_dw',(128,)), kernel=(3,3), stride=(1, 1), dw=True)
conv3       =convolution(input=conv3_dw, w=get_weight('conv3',(128,128,1,1)), b=get_bias('conv3',(128,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv4_dw    =convolution(input=conv3, w=get_weight('conv4_dw',(128,1,3,3)), b=get_bias('conv4_dw',(128,)), kernel=(3,3), stride=(2, 2), dw=True)
conv4       =convolution(input=conv4_dw, w=get_weight('conv4',(256,128,1,1)), b=get_bias('conv4',(256,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv5_dw    =convolution(input=conv4, w=get_weight('conv5_dw',(256,1,3,3)), b=get_bias('conv5_dw',(256,)), kernel=(3,3), stride=(1, 1), dw=True)
conv5       =convolution(input=conv5_dw, w=get_weight('conv5',(256,256,1,1)), b=get_bias('conv5',(256,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv6_dw    =convolution(input=conv5, w=get_weight('conv6_dw',(256,1,3,3)), b=get_bias('conv6_dw',(256,)), kernel=(3,3), stride=(2, 2), dw=True)
conv6       =convolution(input=conv6_dw, w=get_weight('conv6',(512,256,1,1)), b=get_bias('conv6',(512,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv7_dw    =convolution(input=conv6, w=get_weight('conv7_dw',(512,1,3,3)), b=get_bias('conv7_dw',(512,)), kernel=(3,3), stride=(1, 1), dw=True)
conv7       =convolution(input=conv7_dw, w=get_weight('conv7',(512,512,1,1)), b=get_bias('conv7',(512,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv8_dw    =convolution(input=conv7, w=get_weight('conv8_dw',(512,1,3,3)), b=get_bias('conv8_dw',(512,)), kernel=(3,3), stride=(1, 1), dw=True)
conv8       =convolution(input=conv8_dw, w=get_weight('conv8',(512,512,1,1)), b=get_bias('conv8',(512,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv9_dw    =convolution(input=conv8, w=get_weight('conv9_dw',(512,1,3,3)), b=get_bias('conv9_dw',(512,)), kernel=(3,3), stride=(1, 1), dw=True)
conv9       =convolution(input=conv9_dw, w=get_weight('conv9',(512,512,1,1)), b=get_bias('conv9',(512,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv10_dw   =convolution(input=conv9, w=get_weight('conv10_dw',(512,1,3,3)), b=get_bias('conv10_dw',(512,)), kernel=(3,3), stride=(1, 1), dw=True)
conv10      =convolution(input=conv10_dw, w=get_weight('conv10',(512,512,1,1)), b=get_bias('conv10',(512,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv11_dw   =convolution(input=conv10, w=get_weight('conv11_dw',(512,1,3,3)), b=get_bias('conv11_dw',(512,)), kernel=(3,3), stride=(1, 1), dw=True)
conv11      =convolution(input=conv11_dw, w=get_weight('conv11',(512,512,1,1)), b=get_bias('conv11',(512,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv12_dw   =convolution(input=conv11, w=get_weight('conv12_dw',(512,1,3,3)), b=get_bias('conv12_dw',(512,)), kernel=(3,3), stride=(2, 2), dw=True)
conv12      =convolution(input=conv12_dw, w=get_weight('conv12',(1024,512,1,1)), b=get_bias('conv12',(1024,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv13_dw   =convolution(input=conv12, w=get_weight('conv13_dw',(1024,1,3,3)), b=get_bias('conv13_dw',(1024,)), kernel=(3,3), stride=(1, 1), dw=True)
conv13      =convolution(input=conv13_dw, w=get_weight('conv13',(1024,1024,1,1)), b=get_bias('conv13',(1024,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv14_1    =convolution(input=conv13, w=get_weight('conv14_1',(256,1024,1,1)), b=get_bias('conv14_1',(256,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv14_2    =convolution(input=conv14_1, w=get_weight('conv14_2',(512,256,3,3)), b=get_bias('conv14_2',(512,)), kernel=(3,3), stride=(2, 2))
conv15_1    =convolution(input=conv14_2, w=get_weight('conv15_1',(128,512,1,1)), b=get_bias('conv15_1',(128,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv15_2    =convolution(input=conv15_1, w=get_weight('conv15_2',(256,128,3,3)), b=get_bias('conv15_2',(256,)), kernel=(3,3), stride=(2, 2))
conv16_1    =convolution(input=conv15_2, w=get_weight('conv16_1',(128,256,1,1)), b=get_bias('conv16_1',(128,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv16_2    =convolution(input=conv16_1, w=get_weight('conv16_2',(256,128,3,3)), b=get_bias('conv16_2',(256,)), kernel=(3,3), stride=(2, 2))
conv17_1    =convolution(input=conv16_2, w=get_weight('conv17_1',(64,256,1,1)), b=get_bias('conv17_1',(64,)), kernel=(1,1), stride=(1, 1), pad=(0,0))
conv17_2    =convolution(input=conv17_1, w=get_weight('conv17_2',(128,64,3,3)), b=get_bias('conv17_2',(128,)), kernel=(3,3), stride=(2, 2))


conv11_mbox_loc    =convolution(input=conv11, w=get_weight('conv11_mbox_loc',(12,512,1,1)), b=get_bias('conv11_mbox_loc',(12,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv11_mbox_conf    =convolution(input=conv11, w=get_weight('conv11_mbox_conf',(63,512,1,1)), b=get_bias('conv11_mbox_conf',(63,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv13_mbox_loc    =convolution(input=conv13, w=get_weight('conv13_mbox_loc',(24,1024,1,1)), b=get_bias('conv13_mbox_loc',(24,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv13_mbox_conf    =convolution(input=conv13, w=get_weight('conv13_mbox_conf',(126,1024,1,1)), b=get_bias('conv13_mbox_conf',(126,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv14_2_mbox_loc    =convolution(input=conv14_2, w=get_weight('conv14_2_mbox_loc',(24,512,1,1)), b=get_bias('conv14_2_mbox_loc',(24,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv14_2_mbox_conf    =convolution(input=conv14_2, w=get_weight('conv14_2_mbox_conf',(126,512,1,1)), b=get_bias('conv14_2_mbox_conf',(126,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv15_2_mbox_loc    =convolution(input=conv15_2, w=get_weight('conv15_2_mbox_loc',(24,256,1,1)), b=get_bias('conv15_2_mbox_loc',(24,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv15_2_mbox_conf    =convolution(input=conv15_2, w=get_weight('conv15_2_mbox_conf',(126,256,1,1)), b=get_bias('conv15_2_mbox_conf',(126,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv16_2_mbox_loc    =convolution(input=conv16_2, w=get_weight('conv16_2_mbox_loc',(24,256,1,1)), b=get_bias('conv16_2_mbox_loc',(24,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv16_2_mbox_conf    =convolution(input=conv16_2, w=get_weight('conv16_2_mbox_conf',(126,256,1,1)), b=get_bias('conv16_2_mbox_conf',(126,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv17_2_mbox_loc    =convolution(input=conv17_2, w=get_weight('conv17_2_mbox_loc',(24,128,1,1)), b=get_bias('conv17_2_mbox_loc',(24,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
conv17_2_mbox_conf    =convolution(input=conv17_2, w=get_weight('conv17_2_mbox_conf',(126,128,1,1)), b=get_bias('conv17_2_mbox_conf',(126,)), kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)

mbox_conf=get_mbox_conf(conv11_mbox_conf,conv13_mbox_conf,conv14_2_mbox_conf,conv15_2_mbox_conf,conv16_2_mbox_conf,conv17_2_mbox_conf)
mbox_loc=get_mbox_loc(conv11_mbox_loc,conv13_mbox_loc,conv14_2_mbox_loc,conv15_2_mbox_loc,conv16_2_mbox_loc,conv17_2_mbox_loc)
mbox_prior=get_mbox_prior(conv11,conv13,conv14_2,conv15_2,conv16_2,conv17_2)

#conv = nd.Convolution(data=data, weight=W1, bias=b1, kernel=(3,3), num_filter=num_filter)
#conv_relu=nd.relu(conv)
print('end')