import mxnet as mx
import nnvm
import tvm
import numpy as np
import cv2
from PIL import Image
from tvm.contrib import graph_runtime
import nnvm.testing
import nnvm.compiler
import topi
import math

import sys, os, time
import xml_parser
import argparse
from timeit import default_timer as timer

ssd_model = 'PycharmProjects/ssd/ssd_models/mobilenet/ssd_mxnet/deploy_ssd_mobilenet_300_fromcaffe_no_detection'  # input ssd model here
shape = 300
checkpoint = 0
target = 'opencl'

ctx = tvm.context(target, 0)
dshape = (1, 3, shape, shape)
dtype = 'float32'
threshold = 0.01
num_anchor=1917 #8766 #8732
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]


input_folder = '/home/prdcv/Desktop/darknet/test_voc_2007/VOCdevkit/VOC2007/JPEGImages'

DEBUG_MODE = False
ROUND_DECIMAL = 3

if DEBUG_MODE is True:
    debug_file = open("Detail_detection_time_" + str(time.time()) + ".csv", "w")
    header = "total_detection_time, set_data_input_t, get_output_time, multibox_time, mulibox_detection_time\n"
    debug_file.write(header)

# (SIZE, RATIOS, STEP)

default_shape = ((1, 512, 38, 38), (1, 1024, 19, 19), (1, 512, 10, 10),
                 (1, 256, 5, 5), (1, 256, 3, 3), (1, 256, 1, 1))


# end params
# @tvm.register_func
# def tvm_callback_cuda_compile(code):
#     ptx = nvcc.compile_cuda(code, target="ptx")
#     return ptx

def transform_image(image):
    image = np.array(image) - np.array([127.5, 127.5, 127.5])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def transform_image_300(image):
    img = np.array(image) - np.array([123., 117., 104.])
    img = img * 0.007843
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img


def display_plt(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.figsize'] = (10, 10)
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
        plt.gca().text(xmin, ymin - 2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')

    plt.show(block=False)
    plt.pause(0.0001)



def get_multibox_detection_output(np_cls_prob, np_loc_preds, np_anchors, batch_size, num_anchors, num_classes):
    import nnvm.symbol as sym
    cls_prob = sym.Variable("cls_prob")
    loc_preds = sym.Variable("loc_preds")
    anchors = sym.Variable("anchors")
    transform_loc_data, valid_count = sym.multibox_transform_loc(cls_prob=cls_prob, loc_pred=loc_preds,
                                                                 anchor=anchors)
    out = sym.nms(data=transform_loc_data, valid_count=valid_count)

    target = "llvm"
    dtype = "float32"
    ctx = tvm.cpu()
    graph, lib, _ = nnvm.compiler.build(out, target, {"cls_prob": (batch_size, num_classes, num_anchors),
                                                      "loc_preds": (batch_size, num_anchors * 4),
                                                      "anchors": (1, num_anchors, 4)})
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**{"cls_prob": np_cls_prob, "loc_preds": np_loc_preds, "anchors": np_anchors})
    m.run()

    _, out_shape = nnvm.compiler.graph_util.infer_shape(graph, shape={"data": dshape})
    out = m.get_output(0,tvm.nd.empty(tuple(out_shape[0]), dtype))  # output of "mbox_conf_softmax", shape: (1, 21, 8732)
    return out


def get_multibox_detection_output_tvm(np_cls_prob, np_loc_preds, np_anchors, batch_size, num_anchors, num_classes):
    target_cpu = 'llvm'
    ctx = tvm.cpu()

    cls_prob = tvm.placeholder((1, 21, num_anchors), name="cls_prob")
    loc_preds = tvm.placeholder((1, num_anchors * 4), name="loc_preds")
    anchors = tvm.placeholder((1, num_anchors, 4), name="anchors")

    tvm_cls_prob = tvm.nd.array(np_cls_prob.asnumpy().astype(cls_prob.dtype), ctx)
    tvm_loc_preds = tvm.nd.array(np_loc_preds.asnumpy().astype(loc_preds.dtype), ctx)
    tvm_anchors = tvm.nd.array(np_anchors.asnumpy().astype(anchors.dtype), ctx)

    with tvm.target.create(target_cpu):
        out = topi.vision.ssd.multibox_detection(cls_prob, loc_preds, anchors, clip=False, threshold=0.01,
                                                 nms_threshold=0.45,
                                                 force_suppress=False, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)
        s = topi.generic.schedule_multibox_detection(out)

    tvm_out = tvm.nd.array(np.zeros((1, num_anchors, 6)).astype(out.dtype), ctx)
    f = tvm.build(s, [cls_prob, loc_preds, anchors, out], 'llvm')
    f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
    return tvm_out



def get_prior_output(input_data, f, oshape, ctx_multi):
    tvm_input_data = tvm.nd.array(input_data, ctx_multi)
    tvm_out = tvm.nd.array(np.zeros(oshape, dtype=dtype), ctx_multi)
    f(tvm_input_data, tvm_out)
    return tvm_out


def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-imp', '--image_path', help="Image path")
    parser.add_argument('-op', '--out_path', help="XML output path")
    parser.add_argument('-oi', '--out_image', help="Detected images result folder")

    args = parser.parse_args()
    xml_result_folder = ""
    image_output_folder = ""

    OUT_IMAGE = False

    image_path = '/home/prdcv/Desktop/darknet/test_voc_2007/VOCdevkit/VOC2007/JPEGImages'

    if args.out_path is None:
        xml_folder_name = ssd_model.split("/")[-1] + "_xml"
        xml_result_folder = ssd_model.replace(ssd_model.split("/")[-1], xml_folder_name)
    else:
        xml_result_folder = args.out_path

    if os.path.exists(xml_result_folder) is False:
        os.mkdir(xml_result_folder)

    if os.path.exists(image_path) is False:
        print "Please check images path! It was not existed"
        sys.exit()

    if args.out_image is None:
        OUT_IMAGE = False
    else:
        OUT_IMAGE = True
        image_output_folder = args.out_image
        if os.path.exists(image_output_folder) is False:
            os.mkdir(image_output_folder)

    return xml_result_folder, image_path, image_output_folder, OUT_IMAGE


def load_image_data(image_name):

    # using opencv for the convinence
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transform = transform_image_300(cv2.resize(img, (shape, shape)))
    # shape_dict = {'data': img_transform.shape}
    return img, img_transform


def detect_object(nnvm_sym, nnvm_params, img_transform, m, tvm_output):
    set_data_input_t = time.time()
    m.set_input('data', tvm.nd.array(img_transform.astype(dtype)))
    m.run()
    set_data_input_time = round(time.time() - set_data_input_t, ROUND_DECIMAL)
    #print('set_data_input_time: ' + str(set_data_input_time))

    getout_t = time.time()
    tvm_output_0 = m.get_output(0, tvm.nd.empty(tuple(tvm_output[0]),'float32'))  # output of "mbox_conf_softmax", shape: (1, 21, 8732)
    tvm_output_1 = m.get_output(1, tvm.nd.empty(tuple(tvm_output[1]),'float32'))  # output of "mbox_loc" layer, shape: (1, 34928)
    tvm_output_2 = m.get_output(2, tvm.nd.empty(tuple(tvm_output[2]),'float32'))  # output of "broadcast_mul0" layer, shape: (1, 512, 38, 38)
    getout_time = round(time.time() - getout_t, 6)

    #print('getout_time: ' + str(getout_time))

    get_mul_detection_t = time.time()

    final_output = get_multibox_detection_output_tvm(tvm_output_0, tvm_output_1, tvm_output_2, 1, num_anchor, 21)
    get_mul_detection_time = round(time.time() - get_mul_detection_t, ROUND_DECIMAL)
    #print('get_mul_detection_time: ' + str(get_mul_detection_time))
    detect_time = round(set_data_input_time + getout_time + get_mul_detection_time, ROUND_DECIMAL)

    print("run time: "+str(round(set_data_input_time,5))+", getout_time: "+str(round(getout_time,5))+", detection_time: "+ str(round(get_mul_detection_time,5))+", total time: "+str(round(detect_time,5)))
    #print('detect_time: ' + str(detect_time))
    if DEBUG_MODE is True:
        text = str(detect_time) + "," + str(set_data_input_time) + "," + \
               str(getout_time)  + "," + str(get_mul_detection_time) + "\n"
        debug_file.write(text)

    return detect_time, final_output


def isImage(filename):
    isImg = filename.endswith('.png') or filename.endswith('.jpg')
    return isImg


def get_det_value_output(img, raw_det_result, thresh):
    res = []

    for det in raw_det_result:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        text = class_names[cid]
        res.append((cid, score, (xmin, ymin, xmax, ymax)))

    res = sorted(res, key=lambda x: -x[1])
    return res


if __name__ == "__main__":
    xml_result_folder, image_path, image_output_folder, OUT_IMAGE = get_argument()

    mx_sym, args, auxs = mx.model.load_checkpoint(ssd_model, checkpoint)
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, args, auxs)
    print('model compiled.')
    #mx.contrib.quantization.quantize_model()

    ctx = tvm.context(target, 0)
    shape_dict = {'data': dshape}
    with nnvm.compiler.build_config(opt_level=1):
        graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
        
    m_graph = graph_runtime.create(graph, lib, ctx)
    m_graph.set_input(**params)
    _, outshape = nnvm.compiler.graph_util.infer_shape(graph, shape={"data": dshape})

    tvm_ouput_init = []
    for i in range(0, len(outshape)):
        tvm_ouput_init.append(tvm.nd.empty(tuple(outshape[i]), dtype))

    voc_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    class_file = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    for f in os.listdir(image_path):
        image_name = os.path.join(image_path, f)
        #print f
        if os.path.isfile(image_name) and isImage(image_name):
            im_load_start_t = time.time()
            image, img_transform = load_image_data(image_name)
            if image is None or img_transform is None:
                break
            load_img_time = time.time() - im_load_start_t
            detect_time, final_output = detect_object(nnvm_sym, nnvm_params, img_transform, m_graph, outshape)
            res = get_det_value_output(image, final_output.asnumpy()[0], thresh=threshold)
            width=image.shape[1]
            height = image.shape[0]

            for resu in res:

                prob=round(resu[1],4)
                left = max(0, resu[2][0])
                top = max(0, resu[2][1])
                right = min(width, resu[2][2])
                bot = min(height, resu[2][3])

                line = f.replace('.jpg', '') + " " + str(prob) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bot) + '\n'
                class_file[resu[0]] = class_file[resu[0]] + line
                
    for i in range(20):
        file = open(output_folder + 'comp4_det_test_' + voc_name[i]+'.txt', 'w')
        file.write(class_file[i])
        file.close()

    if DEBUG_MODE is True:
        debug_file.close()
