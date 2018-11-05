from mxnet import nd
import mxnet as mx
import cv2
import numpy as np
import tvm
import math


def load_image(image_file):
    origimg = cv2.imread(image_file)
    img = cv2.resize(origimg, (300, 300))
    img = np.array(img) - np.array([123., 117., 104.])
    img = img * 0.007843

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    x = img[np.newaxis, :]
    return x

def get_weight(folder, layer_name, weight_shape):
    weight = np.fromfile(folder+layer_name+'_weight.bin', dtype='float32')
    tt=np.reshape(weight,weight_shape)
    return np.reshape(weight,weight_shape)

def get_bias(folder, layer_name, bias_shape):
    return np.fromfile(folder+layer_name+'_bias.bin', dtype='float32')

def convolution(input, w, b, kernel, stride, pad=(1,1), dw=False, relu=True, no_bias=False, quantize=False, d_scale_prev=1, w_scale=1, d_scale=1):
    if(dw==True):
        group=b.shape[0]
    else:
        group=1
    if(quantize==True):
        #quantize
        weight_int8=np.clip((w*w_scale).round(), -128, 127)
        data_int8=np.clip((input*d_scale_prev).round(), -128, 127)
        #convolution
        #biass=nd.zeros(b.shape)
        top_blob_int32=nd.Convolution(data=nd.array(data_int8), weight=nd.array(weight_int8), bias=nd.zeros(b.shape), kernel=kernel, pad=pad, stride=stride, num_filter=b.shape[0], num_group=group)
        #dequantize
        conv=top_blob_int32/(d_scale_prev*w_scale)
        for i in range(b.shape[0]):
            conv[0][i]=conv[0][i]+b[i]

    else:
        conv=nd.Convolution(data=nd.array(input), weight=nd.array(w), bias=nd.array(b), kernel=kernel, pad=pad, stride=stride, num_filter=b.shape[0], num_group=group)
    if(relu==False):
        return conv.asnumpy()
    else:
        return nd.relu(conv).asnumpy()

def convolution_v1(input, w, b, kernel, stride, pad=(1,1), dw=False, relu=True, no_bias=False, quantize=False, d_scale_prev=1, w_scale=1, d_scale=1):
    if(dw==True):
        group=b.shape[0]
    else:
        group=1
    if(quantize==True):
        #quantize
        weight_new=w*(d_scale/d_scale_prev)
        sp=weight_new.shape
        wk=weight_new.flatten()
        # for i in range(len(wk)):
        #     if(wk[i]==0):
        #         continue
        #     old=wk[i]
        #     int8=round(1/old)
        #     if(int8==0):
        #         continue
        #     wk[i]=(float)(1)/(float)(int8)
        data_int8=np.clip(input.round(), -128, 127)
        bias_int8=np.clip((b*d_scale).round(), -128, 127)
        #data_int8=input.round()
        #bias_new=(b*d_scale).round()
        #convolution
        conv=nd.Convolution(data=nd.array(data_int8), weight=nd.array(wk.reshape(sp)), bias=nd.array(bias_int8), kernel=kernel, pad=pad, stride=stride, num_filter=b.shape[0], num_group=group)
    else:
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
    reshape = concat.reshape(1, 1917, 21)
    transpose=reshape.transpose(0,2,1)
    softmax = nd.SoftmaxActivation(data=nd.array(transpose),mode='channel')
    return softmax

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

def get_detection_out_tvm(np_cls_prob, np_loc_preds, np_anchors, batch_size, num_anchors, num_classes):
    target_cpu = 'llvm'
    ctx = tvm.cpu()
    num_anchors=1917
    cls_prob = tvm.placeholder((1, 21, num_anchors), name="cls_prob")
    loc_preds = tvm.placeholder((1, num_anchors * 4), name="loc_preds")
    anchors = tvm.placeholder((1, num_anchors, 4), name="anchors")

    tvm_cls_prob = tvm.nd.array(np_cls_prob.asnumpy().astype(cls_prob.dtype), ctx)
    tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds.dtype), ctx)
    tvm_anchors = tvm.nd.array(np_anchors.astype(anchors.dtype), ctx)

    import topi
    with tvm.target.create(target_cpu):
        out = topi.vision.ssd.multibox_detection(cls_prob, loc_preds, anchors, clip=False, threshold=0.01,
                                                 nms_threshold=0.45,
                                                 force_suppress=False, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        s = topi.generic.schedule_multibox_detection(out)

    tvm_out = tvm.nd.array(np.zeros((1, num_anchors, 6)).astype(out.dtype), ctx)
    f = tvm.build(s, [cls_prob, loc_preds, anchors, out], 'llvm')
    f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
    return tvm_out

#weight and bias
def mobilenet_quantize(f, image_file):
    data = load_image(image_file)
    conv0 = convolution(input=data, w=get_weight(f,'conv0', (32, 3, 3, 3)), b=get_bias(f,'conv0', (32,)), quantize=False,
                        d_scale_prev=127.033239532, w_scale=27.1343312808, kernel=(3, 3), stride=(2, 2))
    conv1_dw = convolution(input=conv0, w=get_weight(f,'conv1_dw', (32, 1, 3, 3)), b=get_bias(f,'conv1_dw', (32,)),
                           quantize=False, d_scale_prev=10.8546122405, w_scale=23.2944376545, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv1 = convolution(input=conv1_dw, w=get_weight(f,'conv1', (64, 32, 1, 1)), b=get_bias(f,'conv1', (64,)),
                        quantize=False, d_scale_prev=21.9689174791, w_scale=21.9425684997, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv2_dw = convolution(input=conv1, w=get_weight(f,'conv2_dw', (64, 1, 3, 3)), b=get_bias(f,'conv2_dw', (64,)),
                           quantize=False, d_scale_prev=3.37544468118, w_scale=19.6928318641, kernel=(3, 3),
                           stride=(2, 2), dw=True)
    conv2 = convolution(input=conv2_dw, w=get_weight(f,'conv2', (128, 64, 1, 1)), b=get_bias(f,'conv2', (128,)),
                        quantize=False, d_scale_prev=19.1445310806, w_scale=48.8967015262, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv3_dw = convolution(input=conv2, w=get_weight(f,'conv3_dw', (128, 1, 3, 3)), b=get_bias(f,'conv3_dw', (128,)),
                           quantize=False, d_scale_prev=4.71738709348, w_scale=21.4390361661, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv3 = convolution(input=conv3_dw, w=get_weight(f,'conv3', (128, 128, 1, 1)), b=get_bias(f,'conv3', (128,)),
                        quantize=False, d_scale_prev=4.59820614046, w_scale=48.847920493, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv4_dw = convolution(input=conv3, w=get_weight(f,'conv4_dw', (128, 1, 3, 3)), b=get_bias(f,'conv4_dw', (128,)),
                           quantize=False, d_scale_prev=3.42106427837, w_scale=90.5430318409, kernel=(3, 3),
                           stride=(2, 2), dw=True)
    conv4 = convolution(input=conv4_dw, w=get_weight(f,'conv4', (256, 128, 1, 1)), b=get_bias(f,'conv4', (256,)),
                        quantize=False, d_scale_prev=15.5892518815, w_scale=72.4383360454, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv5_dw = convolution(input=conv4, w=get_weight(f,'conv5_dw', (256, 1, 3, 3)), b=get_bias(f,'conv5_dw', (256,)),
                           quantize=False, d_scale_prev=8.66156966607, w_scale=21.3218941238, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv5 = convolution(input=conv5_dw, w=get_weight(f,'conv5', (256, 256, 1, 1)), b=get_bias(f,'conv5', (256,)),
                        quantize=False, d_scale_prev=14.799565296, w_scale=117.769221662, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv6_dw = convolution(input=conv5, w=get_weight(f,'conv6_dw', (256, 1, 3, 3)), b=get_bias(f,'conv6_dw', (256,)),
                           quantize=False, d_scale_prev=9.56802128064, w_scale=37.6473632119, kernel=(3, 3),
                           stride=(2, 2), dw=True)
    conv6 = convolution(input=conv6_dw, w=get_weight(f,'conv6', (512, 256, 1, 1)), b=get_bias(f,'conv6', (512,)),
                        quantize=False, d_scale_prev=15.8153665941, w_scale=119.587535769, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv7_dw = convolution(input=conv6, w=get_weight(f,'conv7_dw', (512, 1, 3, 3)), b=get_bias(f,'conv7_dw', (512,)),
                           quantize=False, d_scale_prev=10.5399075223, w_scale=27.8276505212, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv7 = convolution(input=conv7_dw, w=get_weight(f,'conv7', (512, 512, 1, 1)), b=get_bias(f,'conv7', (512,)),
                        quantize=False, d_scale_prev=13.1339207446, w_scale=145.55121178, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv8_dw = convolution(input=conv7, w=get_weight(f,'conv8_dw', (512, 1, 3, 3)), b=get_bias(f,'conv8_dw', (512,)),
                           quantize=False, d_scale_prev=14.0346975736, w_scale=25.3071462785, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv8 = convolution(input=conv8_dw, w=get_weight(f,'conv8', (512, 512, 1, 1)), b=get_bias(f,'conv8', (512,)),
                        quantize=False, d_scale_prev=14.7030592328, w_scale=91.199442267, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv9_dw = convolution(input=conv8, w=get_weight(f,'conv9_dw', (512, 1, 3, 3)), b=get_bias(f,'conv9_dw', (512,)),
                           quantize=False, d_scale_prev=12.370708375, w_scale=30.5917900825, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv9 = convolution(input=conv9_dw, w=get_weight(f,'conv9', (512, 512, 1, 1)), b=get_bias(f,'conv9', (512,)),
                        quantize=False, d_scale_prev=19.6982374561, w_scale=68.2897934839, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv10_dw = convolution(input=conv9, w=get_weight(f,'conv10_dw', (512, 1, 3, 3)), b=get_bias(f,'conv10_dw', (512,)),
                            quantize=False, d_scale_prev=8.14260696401, w_scale=32.4263879234, kernel=(3, 3),
                            stride=(1, 1), dw=True)
    conv10 = convolution(input=conv10_dw, w=get_weight(f,'conv10', (512, 512, 1, 1)), b=get_bias(f,'conv10', (512,)),
                         quantize=False, d_scale_prev=19.7400047764, w_scale=83.7429915813, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv11_dw = convolution(input=conv10, w=get_weight(f,'conv11_dw', (512, 1, 3, 3)), b=get_bias(f,'conv11_dw', (512,)),
                            quantize=False, d_scale_prev=10.5782605512, w_scale=22.3105033013, kernel=(3, 3),
                            stride=(1, 1), dw=True)
    conv11 = convolution(input=conv11_dw, w=get_weight(f,'conv11', (512, 512, 1, 1)), b=get_bias(f,'conv11', (512,)),
                         quantize=False, d_scale_prev=23.3206331016, w_scale=140.123323673, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv12_dw = convolution(input=conv11, w=get_weight(f,'conv12_dw', (512, 1, 3, 3)), b=get_bias(f,'conv12_dw', (512,)),
                            quantize=False, d_scale_prev=16.7585329697, w_scale=6.2146890293, kernel=(3, 3),
                            stride=(2, 2), dw=True)
    conv12 = convolution(input=conv12_dw, w=get_weight(f,'conv12', (1024, 512, 1, 1)), b=get_bias(f,'conv12', (1024,)),
                         quantize=False, d_scale_prev=22.8822106706, w_scale=133.902477286, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv13_dw = convolution(input=conv12, w=get_weight(f,'conv13_dw', (1024, 1, 3, 3)), b=get_bias(f,'conv13_dw', (1024,)),
                            quantize=False, d_scale_prev=14.7530376997, w_scale=2.69677908927, kernel=(3, 3),
                            stride=(1, 1), dw=True)
    conv13 = convolution(input=conv13_dw, w=get_weight(f,'conv13', (1024, 1024, 1, 1)), b=get_bias(f,'conv13', (1024,)),
                         quantize=False, d_scale_prev=28.6683361259, w_scale=139.46880996, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv14_1 = convolution(input=conv13, w=get_weight(f,'conv14_1', (256, 1024, 1, 1)), b=get_bias(f,'conv14_1', (256,)),
                           quantize=False, d_scale_prev=5.46291092014, w_scale=217.114642238, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv14_2 = convolution(input=conv14_1, w=get_weight(f,'conv14_2', (512, 256, 3, 3)), b=get_bias(f,'conv14_2', (512,)),
                           quantize=False, d_scale_prev=23.6808060224, w_scale=135.036341695, kernel=(3, 3),
                           stride=(2, 2))
    conv15_1 = convolution(input=conv14_2, w=get_weight(f,'conv15_1', (128, 512, 1, 1)), b=get_bias(f,'conv15_1', (128,)),
                           quantize=False, d_scale_prev=26.9081833296, w_scale=363.399615913, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv15_2 = convolution(input=conv15_1, w=get_weight(f,'conv15_2', (256, 128, 3, 3)), b=get_bias(f,'conv15_2', (256,)),
                           quantize=False, d_scale_prev=32.6812935783, w_scale=157.428430349, kernel=(3, 3),
                           stride=(2, 2))
    conv16_1 = convolution(input=conv15_2, w=get_weight(f,'conv16_1', (128, 256, 1, 1)), b=get_bias(f,'conv16_1', (128,)),
                           quantize=False, d_scale_prev=20.7481958129, w_scale=283.463748659, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv16_2 = convolution(input=conv16_1, w=get_weight(f,'conv16_2', (256, 128, 3, 3)), b=get_bias(f,'conv16_2', (256,)),
                           quantize=False, d_scale_prev=39.1085647054, w_scale=140.088296847, kernel=(3, 3),
                           stride=(2, 2))
    conv17_1 = convolution(input=conv16_2, w=get_weight(f,'conv17_1', (64, 256, 1, 1)), b=get_bias(f,'conv17_1', (64,)),
                           quantize=False, d_scale_prev=25.8636239906, w_scale=181.295474825, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv17_2 = convolution(input=conv17_1, w=get_weight(f,'conv17_2', (128, 64, 3, 3)), b=get_bias(f,'conv17_2', (128,)),
                           quantize=False, d_scale_prev=21.368959569, w_scale=98.4925936862, kernel=(3, 3),
                           stride=(2, 2))

    conv11_mbox_loc = convolution(input=conv11, w=get_weight(f,'conv11_mbox_loc', (12, 512, 1, 1)),
                                  b=get_bias(f,'conv11_mbox_loc', (12,)), quantize=False, d_scale_prev=16.7585329697,
                                  w_scale=78.3555422285, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv11_mbox_conf = convolution(input=conv11, w=get_weight(f,'conv11_mbox_conf', (63, 512, 1, 1)),
                                   b=get_bias(f,'conv11_mbox_conf', (63,)), quantize=False, d_scale_prev=16.7585329697,
                                   w_scale=23.8404454112, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv13_mbox_loc = convolution(input=conv13, w=get_weight(f,'conv13_mbox_loc', (24, 1024, 1, 1)),
                                  b=get_bias(f,'conv13_mbox_loc', (24,)), quantize=False, d_scale_prev=5.46291092014,
                                  w_scale=137.15521692, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv13_mbox_conf = convolution(input=conv13, w=get_weight(f,'conv13_mbox_conf', (126, 1024, 1, 1)),
                                   b=get_bias(f,'conv13_mbox_conf', (126,)), quantize=False, d_scale_prev=5.46291092014,
                                   w_scale=20.8723271414, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv14_2_mbox_loc = convolution(input=conv14_2, w=get_weight(f,'conv14_2_mbox_loc', (24, 512, 1, 1)),
                                    b=get_bias(f,'conv14_2_mbox_loc', (24,)), quantize=False, d_scale_prev=26.9081833296,
                                    w_scale=148.255007027, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv14_2_mbox_conf = convolution(input=conv14_2, w=get_weight(f,'conv14_2_mbox_conf', (126, 512, 1, 1)),
                                     b=get_bias(f,'conv14_2_mbox_conf', (126,)), quantize=False,
                                     d_scale_prev=26.9081833296, w_scale=21.0791147728, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)
    conv15_2_mbox_loc = convolution(input=conv15_2, w=get_weight(f,'conv15_2_mbox_loc', (24, 256, 1, 1)),
                                    b=get_bias(f,'conv15_2_mbox_loc', (24,)), quantize=False, d_scale_prev=20.7481958129,
                                    w_scale=156.434490254, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv15_2_mbox_conf = convolution(input=conv15_2, w=get_weight(f,'conv15_2_mbox_conf', (126, 256, 1, 1)),
                                     b=get_bias(f,'conv15_2_mbox_conf', (126,)), quantize=False,
                                     d_scale_prev=20.7481958129, w_scale=22.8982352066, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)
    conv16_2_mbox_loc = convolution(input=conv16_2, w=get_weight(f,'conv16_2_mbox_loc', (24, 256, 1, 1)),
                                    b=get_bias(f,'conv16_2_mbox_loc', (24,)), quantize=False, d_scale_prev=25.8636239906,
                                    w_scale=163.664553431, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv16_2_mbox_conf = convolution(input=conv16_2, w=get_weight(f,'conv16_2_mbox_conf', (126, 256, 1, 1)),
                                     b=get_bias(f,'conv16_2_mbox_conf', (126,)), quantize=False,
                                     d_scale_prev=25.8636239906, w_scale=28.246942679, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)
    conv17_2_mbox_loc = convolution(input=conv17_2, w=get_weight(f,'conv17_2_mbox_loc', (24, 128, 1, 1)),
                                    b=get_bias(f,'conv17_2_mbox_loc', (24,)), quantize=False, d_scale_prev=14.1117994016,
                                    w_scale=262.461097438, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv17_2_mbox_conf = convolution(input=conv17_2, w=get_weight(f,'conv17_2_mbox_conf', (126, 128, 1, 1)),
                                     b=get_bias(f,'conv17_2_mbox_conf', (126,)), quantize=False,
                                     d_scale_prev=14.1117994016, w_scale=41.0664206634, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)

    mbox_conf = get_mbox_conf(conv11_mbox_conf, conv13_mbox_conf, conv14_2_mbox_conf, conv15_2_mbox_conf,
                              conv16_2_mbox_conf, conv17_2_mbox_conf)
    mbox_loc = get_mbox_loc(conv11_mbox_loc, conv13_mbox_loc, conv14_2_mbox_loc, conv15_2_mbox_loc, conv16_2_mbox_loc,
                            conv17_2_mbox_loc)
    mbox_prior = get_mbox_prior(conv11, conv13, conv14_2, conv15_2, conv16_2, conv17_2)

    output = get_detection_out_tvm(mbox_conf, mbox_loc, mbox_prior, 1, 1917, 21)
    return output

#weight and bias from Mxnet
def mobilenet_quantize_test(f, image_file):
    data = load_image(image_file)
    data_input=np.clip((data*127.033239532), -128, 127)
    conv0 = convolution_v1(d_scale=10.8546122405,input=data_input, w=get_weight(f,'conv0', (32, 3, 3, 3)), b=get_bias(f,'conv0', (32,)), quantize=True,
                        d_scale_prev=127.033239532, w_scale=27.1343312808, kernel=(3, 3), stride=(2, 2))
    conv1_dw = convolution_v1(d_scale=21.9689174791,input=conv0, w=get_weight(f,'conv1_dw', (32, 1, 3, 3)), b=get_bias(f,'conv1_dw', (32,)),
                           quantize=True, d_scale_prev=10.8546122405, w_scale=23.2944376545, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv1 = convolution_v1(d_scale=3.37544468118,input=conv1_dw, w=get_weight(f,'conv1', (64, 32, 1, 1)), b=get_bias(f,'conv1', (64,)),
                        quantize=True, d_scale_prev=21.9689174791, w_scale=21.9425684997, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv2_dw = convolution_v1(d_scale=19.1445310806,input=conv1, w=get_weight(f,'conv2_dw', (64, 1, 3, 3)), b=get_bias(f,'conv2_dw', (64,)),
                           quantize=True, d_scale_prev=3.37544468118, w_scale=19.6928318641, kernel=(3, 3),
                           stride=(2, 2), dw=True)
    conv2 = convolution_v1(d_scale=4.71738709348,input=conv2_dw, w=get_weight(f,'conv2', (128, 64, 1, 1)), b=get_bias(f,'conv2', (128,)),
                        quantize=True, d_scale_prev=19.1445310806, w_scale=48.8967015262, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv3_dw = convolution_v1(d_scale=4.59820614046,input=conv2, w=get_weight(f,'conv3_dw', (128, 1, 3, 3)), b=get_bias(f,'conv3_dw', (128,)),
                           quantize=True, d_scale_prev=4.71738709348, w_scale=21.4390361661, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv3 = convolution_v1(d_scale=3.42106427837,input=conv3_dw, w=get_weight(f,'conv3', (128, 128, 1, 1)), b=get_bias(f,'conv3', (128,)),
                        quantize=True, d_scale_prev=4.59820614046, w_scale=48.847920493, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv4_dw = convolution_v1(d_scale=15.5892518815,input=conv3, w=get_weight(f,'conv4_dw', (128, 1, 3, 3)), b=get_bias(f,'conv4_dw', (128,)),
                           quantize=True, d_scale_prev=3.42106427837, w_scale=90.5430318409, kernel=(3, 3),
                           stride=(2, 2), dw=True)
    conv4 = convolution_v1(d_scale=8.66156966607,input=conv4_dw, w=get_weight(f,'conv4', (256, 128, 1, 1)), b=get_bias(f,'conv4', (256,)),
                        quantize=True, d_scale_prev=15.5892518815, w_scale=72.4383360454, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv5_dw = convolution_v1(d_scale=14.799565296,input=conv4, w=get_weight(f,'conv5_dw', (256, 1, 3, 3)), b=get_bias(f,'conv5_dw', (256,)),
                           quantize=True, d_scale_prev=8.66156966607, w_scale=21.3218941238, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv5 = convolution_v1(d_scale=9.56802128064,input=conv5_dw, w=get_weight(f,'conv5', (256, 256, 1, 1)), b=get_bias(f,'conv5', (256,)),
                        quantize=True, d_scale_prev=14.799565296, w_scale=117.769221662, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv6_dw = convolution_v1(d_scale=15.8153665941,input=conv5, w=get_weight(f,'conv6_dw', (256, 1, 3, 3)), b=get_bias(f,'conv6_dw', (256,)),
                           quantize=True, d_scale_prev=9.56802128064, w_scale=37.6473632119, kernel=(3, 3),
                           stride=(2, 2), dw=True)
    conv6 = convolution_v1(d_scale=10.5399075223,input=conv6_dw, w=get_weight(f,'conv6', (512, 256, 1, 1)), b=get_bias(f,'conv6', (512,)),
                        quantize=True, d_scale_prev=15.8153665941, w_scale=119.587535769, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv7_dw = convolution_v1(d_scale=13.1339207446,input=conv6, w=get_weight(f,'conv7_dw', (512, 1, 3, 3)), b=get_bias(f,'conv7_dw', (512,)),
                           quantize=True, d_scale_prev=10.5399075223, w_scale=27.8276505212, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv7 = convolution_v1(d_scale=14.0346975736,input=conv7_dw, w=get_weight(f,'conv7', (512, 512, 1, 1)), b=get_bias(f,'conv7', (512,)),
                        quantize=True, d_scale_prev=13.1339207446, w_scale=145.55121178, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv8_dw = convolution_v1(d_scale=14.7030592328,input=conv7, w=get_weight(f,'conv8_dw', (512, 1, 3, 3)), b=get_bias(f,'conv8_dw', (512,)),
                           quantize=True, d_scale_prev=14.0346975736, w_scale=25.3071462785, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv8 = convolution_v1(d_scale=12.370708375,input=conv8_dw, w=get_weight(f,'conv8', (512, 512, 1, 1)), b=get_bias(f,'conv8', (512,)),
                        quantize=True, d_scale_prev=14.7030592328, w_scale=91.199442267, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv9_dw = convolution_v1(d_scale=19.6982374561,input=conv8, w=get_weight(f,'conv9_dw', (512, 1, 3, 3)), b=get_bias(f,'conv9_dw', (512,)),
                           quantize=True, d_scale_prev=12.370708375, w_scale=30.5917900825, kernel=(3, 3),
                           stride=(1, 1), dw=True)
    conv9 = convolution_v1(d_scale=8.14260696401,input=conv9_dw, w=get_weight(f,'conv9', (512, 512, 1, 1)), b=get_bias(f,'conv9', (512,)),
                        quantize=True, d_scale_prev=19.6982374561, w_scale=68.2897934839, kernel=(1, 1), stride=(1, 1),
                        pad=(0, 0))
    conv10_dw = convolution_v1(d_scale=19.7400047764,input=conv9, w=get_weight(f,'conv10_dw', (512, 1, 3, 3)), b=get_bias(f,'conv10_dw', (512,)),
                            quantize=True, d_scale_prev=8.14260696401, w_scale=32.4263879234, kernel=(3, 3),
                            stride=(1, 1), dw=True)
    conv10 = convolution_v1(d_scale=10.5782605512,input=conv10_dw, w=get_weight(f,'conv10', (512, 512, 1, 1)), b=get_bias(f,'conv10', (512,)),
                         quantize=True, d_scale_prev=19.7400047764, w_scale=83.7429915813, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv11_dw = convolution_v1(d_scale=23.3206331016,input=conv10, w=get_weight(f,'conv11_dw', (512, 1, 3, 3)), b=get_bias(f,'conv11_dw', (512,)),
                            quantize=True, d_scale_prev=10.5782605512, w_scale=22.3105033013, kernel=(3, 3),
                            stride=(1, 1), dw=True)
    conv11 = convolution_v1(d_scale=16.7585329697,input=conv11_dw, w=get_weight(f,'conv11', (512, 512, 1, 1)), b=get_bias(f,'conv11', (512,)),
                         quantize=True, d_scale_prev=23.3206331016, w_scale=140.123323673, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv12_dw = convolution_v1(d_scale=22.8822106706,input=conv11, w=get_weight(f,'conv12_dw', (512, 1, 3, 3)), b=get_bias(f,'conv12_dw', (512,)),
                            quantize=True, d_scale_prev=16.7585329697, w_scale=6.2146890293, kernel=(3, 3),
                            stride=(2, 2), dw=True)
    conv12 = convolution_v1(d_scale=14.7530376997,input=conv12_dw, w=get_weight(f,'conv12', (1024, 512, 1, 1)), b=get_bias(f,'conv12', (1024,)),
                         quantize=True, d_scale_prev=22.8822106706, w_scale=133.902477286, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv13_dw = convolution_v1(d_scale=28.6683361259,input=conv12, w=get_weight(f,'conv13_dw', (1024, 1, 3, 3)), b=get_bias(f,'conv13_dw', (1024,)),
                            quantize=True, d_scale_prev=14.7530376997, w_scale=2.69677908927, kernel=(3, 3),
                            stride=(1, 1), dw=True)
    conv13 = convolution_v1(d_scale=5.46291092014,input=conv13_dw, w=get_weight(f,'conv13', (1024, 1024, 1, 1)), b=get_bias(f,'conv13', (1024,)),
                         quantize=True, d_scale_prev=28.6683361259, w_scale=139.46880996, kernel=(1, 1), stride=(1, 1),
                         pad=(0, 0))
    conv14_1 = convolution_v1(d_scale=23.6808060224,input=conv13, w=get_weight(f,'conv14_1', (256, 1024, 1, 1)), b=get_bias(f,'conv14_1', (256,)),
                           quantize=True, d_scale_prev=5.46291092014, w_scale=217.114642238, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv14_2 = convolution_v1(d_scale=26.9081833296,input=conv14_1, w=get_weight(f,'conv14_2', (512, 256, 3, 3)), b=get_bias(f,'conv14_2', (512,)),
                           quantize=True, d_scale_prev=23.6808060224, w_scale=135.036341695, kernel=(3, 3),
                           stride=(2, 2))
    conv15_1 = convolution_v1(d_scale=32.6812935783,input=conv14_2, w=get_weight(f,'conv15_1', (128, 512, 1, 1)), b=get_bias(f,'conv15_1', (128,)),
                           quantize=True, d_scale_prev=26.9081833296, w_scale=363.399615913, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv15_2 = convolution_v1(d_scale=20.7481958129,input=conv15_1, w=get_weight(f,'conv15_2', (256, 128, 3, 3)), b=get_bias(f,'conv15_2', (256,)),
                           quantize=True, d_scale_prev=32.6812935783, w_scale=157.428430349, kernel=(3, 3),
                           stride=(2, 2))
    conv16_1 = convolution_v1(d_scale=39.1085647054,input=conv15_2, w=get_weight(f,'conv16_1', (128, 256, 1, 1)), b=get_bias(f,'conv16_1', (128,)),
                           quantize=True, d_scale_prev=20.7481958129, w_scale=283.463748659, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv16_2 = convolution_v1(d_scale=25.8636239906,input=conv16_1, w=get_weight(f,'conv16_2', (256, 128, 3, 3)), b=get_bias(f,'conv16_2', (256,)),
                           quantize=True, d_scale_prev=39.1085647054, w_scale=140.088296847, kernel=(3, 3),
                           stride=(2, 2))
    conv17_1 = convolution_v1(d_scale=21.368959569,input=conv16_2, w=get_weight(f,'conv17_1', (64, 256, 1, 1)), b=get_bias(f,'conv17_1', (64,)),
                           quantize=True, d_scale_prev=25.8636239906, w_scale=181.295474825, kernel=(1, 1),
                           stride=(1, 1), pad=(0, 0))
    conv17_2 = convolution_v1(d_scale=14.1117994016,input=conv17_1, w=get_weight(f,'conv17_2', (128, 64, 3, 3)), b=get_bias(f,'conv17_2', (128,)),
                           quantize=True, d_scale_prev=21.368959569, w_scale=98.4925936862, kernel=(3, 3),
                           stride=(2, 2))

    conv11_mbox_loc = convolution_v1(d_scale=1,input=conv11, w=get_weight(f,'conv11_mbox_loc', (12, 512, 1, 1)),
                                  b=get_bias(f,'conv11_mbox_loc', (12,)), quantize=True, d_scale_prev=16.7585329697,
                                  w_scale=78.3555422285, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv11_mbox_conf = convolution_v1(d_scale=1,input=conv11, w=get_weight(f,'conv11_mbox_conf', (63, 512, 1, 1)),
                                   b=get_bias(f,'conv11_mbox_conf', (63,)), quantize=True, d_scale_prev=16.7585329697,
                                   w_scale=23.8404454112, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv13_mbox_loc = convolution_v1(d_scale=1,input=conv13, w=get_weight(f,'conv13_mbox_loc', (24, 1024, 1, 1)),
                                  b=get_bias(f,'conv13_mbox_loc', (24,)), quantize=True, d_scale_prev=5.46291092014,
                                  w_scale=137.15521692, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv13_mbox_conf = convolution_v1(d_scale=1,input=conv13, w=get_weight(f,'conv13_mbox_conf', (126, 1024, 1, 1)),
                                   b=get_bias(f,'conv13_mbox_conf', (126,)), quantize=True, d_scale_prev=5.46291092014,
                                   w_scale=20.8723271414, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv14_2_mbox_loc = convolution_v1(d_scale=1,input=conv14_2, w=get_weight(f,'conv14_2_mbox_loc', (24, 512, 1, 1)),
                                    b=get_bias(f,'conv14_2_mbox_loc', (24,)), quantize=True, d_scale_prev=26.9081833296,
                                    w_scale=148.255007027, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv14_2_mbox_conf = convolution_v1(d_scale=1,input=conv14_2, w=get_weight(f,'conv14_2_mbox_conf', (126, 512, 1, 1)),
                                     b=get_bias(f,'conv14_2_mbox_conf', (126,)), quantize=True,
                                     d_scale_prev=26.9081833296, w_scale=21.0791147728, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)
    conv15_2_mbox_loc = convolution_v1(d_scale=1,input=conv15_2, w=get_weight(f,'conv15_2_mbox_loc', (24, 256, 1, 1)),
                                    b=get_bias(f,'conv15_2_mbox_loc', (24,)), quantize=True, d_scale_prev=20.7481958129,
                                    w_scale=156.434490254, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv15_2_mbox_conf = convolution_v1(d_scale=1,input=conv15_2, w=get_weight(f,'conv15_2_mbox_conf', (126, 256, 1, 1)),
                                     b=get_bias(f,'conv15_2_mbox_conf', (126,)), quantize=True,
                                     d_scale_prev=20.7481958129, w_scale=22.8982352066, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)
    conv16_2_mbox_loc = convolution_v1(d_scale=1,input=conv16_2, w=get_weight(f,'conv16_2_mbox_loc', (24, 256, 1, 1)),
                                    b=get_bias(f,'conv16_2_mbox_loc', (24,)), quantize=True, d_scale_prev=25.8636239906,
                                    w_scale=163.664553431, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv16_2_mbox_conf = convolution_v1(d_scale=1,input=conv16_2, w=get_weight(f,'conv16_2_mbox_conf', (126, 256, 1, 1)),
                                     b=get_bias(f,'conv16_2_mbox_conf', (126,)), quantize=True,
                                     d_scale_prev=25.8636239906, w_scale=28.246942679, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)
    conv17_2_mbox_loc = convolution_v1(d_scale=1,input=conv17_2, w=get_weight(f,'conv17_2_mbox_loc', (24, 128, 1, 1)),
                                    b=get_bias(f,'conv17_2_mbox_loc', (24,)), quantize=True, d_scale_prev=14.1117994016,
                                    w_scale=262.461097438, kernel=(1, 1), stride=(1, 1), pad=(0, 0), relu=False)
    conv17_2_mbox_conf = convolution_v1(d_scale=1,input=conv17_2, w=get_weight(f,'conv17_2_mbox_conf', (126, 128, 1, 1)),
                                     b=get_bias(f,'conv17_2_mbox_conf', (126,)), quantize=True,
                                     d_scale_prev=14.1117994016, w_scale=41.0664206634, kernel=(1, 1), stride=(1, 1),
                                     pad=(0, 0), relu=False)

    mbox_conf = get_mbox_conf(conv11_mbox_conf, conv13_mbox_conf, conv14_2_mbox_conf, conv15_2_mbox_conf,
                              conv16_2_mbox_conf, conv17_2_mbox_conf)
    mbox_loc = get_mbox_loc(conv11_mbox_loc, conv13_mbox_loc, conv14_2_mbox_loc, conv15_2_mbox_loc, conv16_2_mbox_loc,
                            conv17_2_mbox_loc)
    mbox_prior = get_mbox_prior(conv11, conv13, conv14_2, conv15_2, conv16_2, conv17_2)

    output = get_detection_out_tvm(mbox_conf, mbox_loc, mbox_prior, 1, 1917, 21)
    return output

