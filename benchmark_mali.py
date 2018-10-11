import mxnet as mx
import nnvm
import tvm
import numpy as np
from PIL import Image
from tvm.contrib import graph_runtime
import nnvm.compiler
import cv2
import topi
import time
import math
from nnvm import testing


ssd_model='/home/firefly/AVC/deploy_ssd_mobilenet_300_fromcaffe_no_prior_detection'
shape=300
checkpoint=0
num_anchor=1917
target=tvm.target.mali()
#target='opencl'
ctx = tvm.cl(0)
dshape = (1, 3, shape, shape)
dtype = 'float32'
threshold=0.25
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                             edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
        #print(str(text)+", Score: "+str(score))
    plt.show()

def get_multibox_detection_output_tvm(np_cls_prob, np_loc_preds, np_anchors, batch_size, num_anchors, num_classes):
    target_cpu = 'llvm'
    ctx = tvm.cpu()

    cls_prob = tvm.placeholder((1, 21, num_anchors), name="cls_prob")
    loc_preds = tvm.placeholder((1, num_anchors * 4), name="loc_preds")
    anchors = tvm.placeholder((1, num_anchors, 4), name="anchors")

    tvm_cls_prob = tvm.nd.array(np_cls_prob.asnumpy().astype(cls_prob.dtype), ctx)
    tvm_loc_preds = tvm.nd.array(np_loc_preds.asnumpy().astype(loc_preds.dtype), ctx)
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



def get_multibox_prior_tvm(priorBox_1,priorBox_2,priorBox_3,priorBox_4,priorBox_5,priorBox_6):

    output_priorBox_1 = get_prior_output(priorBox_1, sizes=(0.2,), ratios=(1.0, 2.0, 0.5), steps=(-0.003333, -0.003333))
    output_priorBox_2 = get_prior_output(priorBox_2, sizes=(0.35, 0.41833), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_3 = get_prior_output(priorBox_3, sizes=(0.5, 0.570088), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_4 = get_prior_output(priorBox_4, sizes=(0.65, 0.72111), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_5 = get_prior_output(priorBox_5, sizes=(0.8, 0.87178), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))
    output_priorBox_6 = get_prior_output(priorBox_6, sizes=(0.95, 0.974679), ratios=(1.0, 2.0, 0.5, 3.0, 0.333333333333), steps=(-0.003333, -0.003333))

    flatten1=output_priorBox_1.asnumpy().reshape(1,output_priorBox_1.shape[1]*output_priorBox_1.shape[2])
    flatten2=output_priorBox_2.asnumpy().reshape(1,output_priorBox_2.shape[1]*output_priorBox_2.shape[2])
    flatten3=output_priorBox_3.asnumpy().reshape(1,output_priorBox_3.shape[1]*output_priorBox_3.shape[2])
    flatten4=output_priorBox_4.asnumpy().reshape(1,output_priorBox_4.shape[1]*output_priorBox_4.shape[2])
    flatten5=output_priorBox_5.asnumpy().reshape(1,output_priorBox_5.shape[1]*output_priorBox_5.shape[2])
    flatten6=output_priorBox_6.asnumpy().reshape(1,output_priorBox_6.shape[1]*output_priorBox_6.shape[2])
    concat=np.concatenate((flatten1,flatten2,flatten3,flatten4,flatten5,flatten6),axis=1)
    multibox_prior=concat.reshape(1,1917,4)
    return multibox_prior

def get_prior_output(input_data, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    target_cpu = 'llvm'
    ctx = tvm.cpu()
    dshape=input_data.shape
    data = tvm.placeholder(dshape, name="data")
    dtype = data.dtype


    in_height = data.shape[2].value
    in_width = data.shape[3].value
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


    with tvm.target.create(target_cpu):
        out = topi.vision.ssd.multibox_prior(data, sizes, ratios, steps, offsets, clip)
        s = topi.generic.schedule_multibox_prior(out)

    tvm_input_data = tvm.nd.array(input_data, ctx)
    tvm_out = tvm.nd.array(np.zeros(oshape, dtype=dtype), ctx)
    f = tvm.build(s, [data, out], target_cpu)
    f(tvm_input_data, tvm_out)
    return tvm_out

def build_network(model):
    shape=224
    dshape=(1, 3,224,224)
    if model == 'vgg16':
        net, params = testing.vgg.get_workload(num_layers=16,
            batch_size=1, image_shape=(3, 224, 224), dtype=dtype)
    elif model == 'resnet18':
        net, params = testing.resnet.get_workload(num_layers=18,
            batch_size=1, image_shape=(3, 224, 224), dtype=dtype)
    elif model == 'mobilenet':
        net, params = testing.mobilenet.get_workload(
            batch_size=1, image_shape=(3, 224, 224), dtype=dtype)
    elif model == 'mobilenet_v2':
        net, params = testing.mobilenet_v2.get_workload(
            batch_size=1, image_shape=(3, 224, 224), dtype=dtype)
    elif model == 'squeezenet':
        net, params = testing.squeezenet.get_workload(
            batch_size=1, image_shape=(3, 224, 224), dtype=dtype)
    elif model == 'mobilenet_ssd':
        dshape=(1,3,300,300)
        mx_sym, args, auxs = mx.model.load_checkpoint(ssd_model, checkpoint)
        net, params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    else:
        raise ValueError('no benchmark prepared for {}.'.format(model))

    # compile

    with nnvm.compiler.build_config(opt_level=2):
        graph, lib, params = nnvm.compiler.build(net, target,  shape={'data': dshape}, params=params)

    return graph, lib, params

model='mobilenet_v2'
graph, lib, params= build_network(model)
print('model: '+model)


# Execute the portable graph on TVM
m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)

data_tvm = tvm.nd.array((np.random.uniform(size=dshape).astype('float32')))
m.set_input('data', data_tvm)

num_warmup = 10
num_test   = 60
if model == 'mobilenet': # mobilenet is fast, need more runs for stable measureament
    num_warmup *= 5
    num_test   *= 5

#perform some warm up runs
print("warm up..")
warm_up_timer = m.module.time_evaluator("run", ctx, num_warmup)
warm_up_timer()

# test
print("test..")
ftimer = m.module.time_evaluator("run", ctx, num_test)
prof_res = ftimer()
print(prof_res)


#visualize for mobilenet_ssd only
if model=='mobilenet_ssdd':
    m.run()
    _, outshape = nnvm.compiler.graph_util.infer_shape(graph, shape={"data": dshape})

    begin2 = time.time()
    tvm_output_0 = m.get_output(0, tvm.nd.empty(tuple(outshape[0]),'float32'))  # shape: (1, 21, 1917)
    tvm_output_1 = m.get_output(1, tvm.nd.empty(tuple(outshape[1]),'float32'))  #shape: (1, 7668)
    tvm_output_2 = m.get_output(2, tvm.nd.empty(tuple(outshape[2]),'float32'))  # shape: (1, 1917, 4)
    tvm_output_3 = m.get_output(3, tvm.nd.empty(tuple(outshape[3]), dtype)) # output of "relu7" layer, shape: (1, 1024, 19, 19)
    tvm_output_4 = m.get_output(4, tvm.nd.empty(tuple(outshape[4]), dtype)) # output of "conv6_2_relu" layer, shape: (1, 512, 10, 10)
    tvm_output_5 = m.get_output(5, tvm.nd.empty(tuple(outshape[5]), dtype)) # output of "conv7_2_relu" layer, shape: (1, 256, 5, 5)  178
    tvm_output_6 = m.get_output(6, tvm.nd.empty(tuple(outshape[6]), dtype)) # output of "conv8_2_relu" layer, shape: (1, 256, 3, 3)
    tvm_output_7 = m.get_output(7, tvm.nd.empty(tuple(outshape[7]), dtype))
    end4 = time.time()
    print('get_output time: '+str(end4-begin2))

    begin5 = time.time()
    multibox_loc_preds= get_multibox_prior_tvm(tvm_output_2,tvm_output_3,tvm_output_4,tvm_output_5,tvm_output_6,tvm_output_7)
    final_output = get_multibox_detection_output_tvm(tvm_output_0, tvm_output_1, multibox_loc_preds, 1, num_anchor, 21)
    end5 = time.time()
    print('post-processing time: '+str(end5-begin5))

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display(image, final_output.asnumpy()[0], thresh=threshold)
