import mxnet as mx
import cv2
import time
import mobilenet_ssd_rebuild
import os

ctx = mx.cpu()
shape=300
image_file='/home/prdcv/PycharmProjects/gvh205/others/images/dog.jpg'
num_anchor=1917
threshold=0.25
class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

input_folder = '/home/prdcv/Desktop/darknet/test_voc_2007/VOCdevkit/VOC2007/JPEGImages'
#input_folder = '/home/prdcv/PycharmProjects/gvh205/quantization/imgs'
#output_folder='/home/prdcv/PycharmProjects/gvh205/quantization/predict/'


def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    count=0
    for det in out:

        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        count += 1
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
    print("total object detected: "+str(count))
    plt.show()

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

def detect_single(img_path, weight_bias_folder, test=False):
    if(test==False):
        output = mobilenet_ssd_rebuild.mobilenet_quantize(weight_bias_folder, img_path)
    else:
        output = mobilenet_ssd_rebuild.mobilenet_quantize_test(weight_bias_folder, img_path)

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display(image, output.asnumpy()[0], thresh=threshold)
    return

def detect_multi(img_folder, weight_bias_folder, output_folder):
    class_file = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    allfiles = os.listdir(img_folder)
    allfiles = sorted(allfiles)

    for f in allfiles:
        image_name = os.path.join(img_folder, f)
        print f
        if os.path.isfile(image_name) and isImage(image_name):
            output = mobilenet_ssd_rebuild.mobilenet_quantize(weight_bias_folder, image_name)
            image = cv2.imread(image_name)
            det_value = get_det_value_output(image, output.asnumpy()[0], thresh=threshold)
            width=image.shape[1]
            height = image.shape[0]

            for obj in det_value:
                prob=round(obj[1],4)
                left = max(0, obj[2][0])
                top = max(0, obj[2][1])
                right = min(width, obj[2][2])
                bot = min(height, obj[2][3])

                line = f.replace('.jpg', '') + " " + str(prob) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bot) + '\n'
                class_file[obj[0]] = class_file[obj[0]] + line

    for i in range(20):
        file = open(output_folder + 'comp4_det_test_' + class_names[i]+'.txt', 'w')
        file.write(class_file[i])
        file.close()

def detect_multi_test(img_folder, weight_bias_folder, output_folder):
    class_file = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

    allfiles = os.listdir(img_folder)
    allfiles = sorted(allfiles)

    for f in allfiles:
        image_name = os.path.join(img_folder, f)
        print f
        if os.path.isfile(image_name) and isImage(image_name):
            output = mobilenet_ssd_rebuild.mobilenet_quantize_test(weight_bias_folder, image_name)
            image = cv2.imread(image_name)
            det_value = get_det_value_output(image, output.asnumpy()[0], thresh=threshold)
            width=image.shape[1]
            height = image.shape[0]

            for obj in det_value:
                prob=round(obj[1],4)
                left = max(0, obj[2][0])
                top = max(0, obj[2][1])
                right = min(width, obj[2][2])
                bot = min(height, obj[2][3])

                line = f.replace('.jpg', '') + " " + str(prob) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bot) + '\n'
                class_file[obj[0]] = class_file[obj[0]] + line

    for i in range(20):
        file = open(output_folder + 'comp4_det_test_' + class_names[i]+'.txt', 'w')
        file.write(class_file[i])
        file.close()

begin=time.time()
#detect_single(image_file,'/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/fp32_origin/')
#detect_single(image_file,'/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/ncnn/fp32_test/')
detect_single(image_file,'/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/mxnet/fp32_origin/',test=False)

#detect_multi(input_folder, '/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/ncnn/fp32_origin/','/home/prdcv/PycharmProjects/gvh205/quantization/predict_ncnn_no_quantize/')
#detect_multi_test(input_folder, '/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/ncnn_fp32_origin/','/home/prdcv/PycharmProjects/gvh205/quantization/predict_ncnn_test/')
#detect_multi(input_folder, '/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/fp32_origin/','/home/prdcv/PycharmProjects/gvh205/quantization/predict_mxnet/')
#detect_multi_test(input_folder, '/home/prdcv/PycharmProjects/gvh205/Mobilenet_bin/fp32_origin/','/home/prdcv/PycharmProjects/gvh205/quantization/predict_mxnet_test/')
print("processing time: "+str(time.time()-begin))
print('end')




#
# data        =load_image(image_file)
# conv0       =convolution(input=data, w=get_weight('conv0',(32,3,3,3)),b=get_bias('conv0',(32,)),quantize=True, d_scale_prev= 127.033239532, w_scale=27.1343312808, kernel=(3,3), stride=(2, 2))
# conv1_dw    =convolution(input=conv0, w=get_weight('conv1_dw',(32,1,3,3)), b=get_bias('conv1_dw',(32,)),quantize=True, d_scale_prev= 10.8546122405, w_scale=23.2944376545, kernel=(3,3), stride=(1, 1), dw=True)
# conv1       =convolution(input=conv1_dw, w=get_weight('conv1',(64,32,1,1)), b=get_bias('conv1',(64,)),quantize=False, d_scale_prev= 21.9689174791, w_scale=21.9425684997, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv2_dw    =convolution(input=conv1, w=get_weight('conv2_dw',(64,1,3,3)), b=get_bias('conv2_dw',(64,)),quantize=True, d_scale_prev= 3.37544468118, w_scale=19.6928318641, kernel=(3,3), stride=(2, 2), dw=True)
# conv2       =convolution(input=conv2_dw, w=get_weight('conv2',(128,64,1,1)), b=get_bias('conv2',(128,)),quantize=True, d_scale_prev= 19.1445310806, w_scale=48.8967015262, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv3_dw    =convolution(input=conv2, w=get_weight('conv3_dw',(128,1,3,3)), b=get_bias('conv3_dw',(128,)),quantize=True, d_scale_prev= 4.71738709348, w_scale=21.4390361661, kernel=(3,3), stride=(1, 1), dw=True)
# conv3       =convolution(input=conv3_dw, w=get_weight('conv3',(128,128,1,1)), b=get_bias('conv3',(128,)),quantize=True, d_scale_prev= 4.59820614046, w_scale=48.847920493, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv4_dw    =convolution(input=conv3, w=get_weight('conv4_dw',(128,1,3,3)), b=get_bias('conv4_dw',(128,)),quantize=True, d_scale_prev= 3.42106427837, w_scale=90.5430318409, kernel=(3,3), stride=(2, 2), dw=True)
# conv4       =convolution(input=conv4_dw, w=get_weight('conv4',(256,128,1,1)), b=get_bias('conv4',(256,)),quantize=True, d_scale_prev= 15.5892518815, w_scale=72.4383360454, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv5_dw    =convolution(input=conv4, w=get_weight('conv5_dw',(256,1,3,3)), b=get_bias('conv5_dw',(256,)),quantize=True, d_scale_prev= 8.66156966607, w_scale=21.3218941238, kernel=(3,3), stride=(1, 1), dw=True)
# conv5       =convolution(input=conv5_dw, w=get_weight('conv5',(256,256,1,1)), b=get_bias('conv5',(256,)),quantize=True, d_scale_prev= 14.799565296, w_scale=117.769221662, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv6_dw    =convolution(input=conv5, w=get_weight('conv6_dw',(256,1,3,3)), b=get_bias('conv6_dw',(256,)),quantize=True, d_scale_prev= 9.56802128064, w_scale=37.6473632119, kernel=(3,3), stride=(2, 2), dw=True)
# conv6       =convolution(input=conv6_dw, w=get_weight('conv6',(512,256,1,1)), b=get_bias('conv6',(512,)),quantize=True, d_scale_prev= 15.8153665941, w_scale=119.587535769, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv7_dw    =convolution(input=conv6, w=get_weight('conv7_dw',(512,1,3,3)), b=get_bias('conv7_dw',(512,)),quantize=True, d_scale_prev= 10.5399075223, w_scale=27.8276505212, kernel=(3,3), stride=(1, 1), dw=True)
# conv7       =convolution(input=conv7_dw, w=get_weight('conv7',(512,512,1,1)), b=get_bias('conv7',(512,)),quantize=True, d_scale_prev= 13.1339207446, w_scale=145.55121178, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv8_dw    =convolution(input=conv7, w=get_weight('conv8_dw',(512,1,3,3)), b=get_bias('conv8_dw',(512,)),quantize=True, d_scale_prev= 14.0346975736, w_scale=25.3071462785, kernel=(3,3), stride=(1, 1), dw=True)
# conv8       =convolution(input=conv8_dw, w=get_weight('conv8',(512,512,1,1)), b=get_bias('conv8',(512,)),quantize=True, d_scale_prev= 14.7030592328, w_scale=91.199442267, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv9_dw    =convolution(input=conv8, w=get_weight('conv9_dw',(512,1,3,3)), b=get_bias('conv9_dw',(512,)),quantize=True, d_scale_prev= 12.370708375, w_scale=30.5917900825, kernel=(3,3), stride=(1, 1), dw=True)
# conv9       =convolution(input=conv9_dw, w=get_weight('conv9',(512,512,1,1)), b=get_bias('conv9',(512,)),quantize=True, d_scale_prev= 19.6982374561, w_scale=68.2897934839, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv10_dw   =convolution(input=conv9, w=get_weight('conv10_dw',(512,1,3,3)), b=get_bias('conv10_dw',(512,)),quantize=True, d_scale_prev= 8.14260696401, w_scale=32.4263879234, kernel=(3,3), stride=(1, 1), dw=True)
# conv10      =convolution(input=conv10_dw, w=get_weight('conv10',(512,512,1,1)), b=get_bias('conv10',(512,)),quantize=True, d_scale_prev= 19.7400047764, w_scale=83.7429915813, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv11_dw   =convolution(input=conv10, w=get_weight('conv11_dw',(512,1,3,3)), b=get_bias('conv11_dw',(512,)),quantize=True, d_scale_prev= 10.5782605512, w_scale=22.3105033013, kernel=(3,3), stride=(1, 1), dw=True)
# conv11      =convolution(input=conv11_dw, w=get_weight('conv11',(512,512,1,1)), b=get_bias('conv11',(512,)),quantize=True, d_scale_prev= 23.3206331016, w_scale=140.123323673, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv12_dw   =convolution(input=conv11, w=get_weight('conv12_dw',(512,1,3,3)), b=get_bias('conv12_dw',(512,)),quantize=True, d_scale_prev= 16.7585329697, w_scale=6.2146890293, kernel=(3,3), stride=(2, 2), dw=True)
# conv12      =convolution(input=conv12_dw, w=get_weight('conv12',(1024,512,1,1)), b=get_bias('conv12',(1024,)),quantize=True, d_scale_prev= 22.8822106706, w_scale=133.902477286, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv13_dw   =convolution(input=conv12, w=get_weight('conv13_dw',(1024,1,3,3)), b=get_bias('conv13_dw',(1024,)),quantize=True, d_scale_prev= 14.7530376997, w_scale=2.69677908927, kernel=(3,3), stride=(1, 1), dw=True)
# conv13      =convolution(input=conv13_dw, w=get_weight('conv13',(1024,1024,1,1)), b=get_bias('conv13',(1024,)),quantize=True, d_scale_prev= 28.6683361259, w_scale=139.46880996, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv14_1    =convolution(input=conv13, w=get_weight('conv14_1',(256,1024,1,1)), b=get_bias('conv14_1',(256,)),quantize=True, d_scale_prev= 5.46291092014, w_scale=217.114642238, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv14_2    =convolution(input=conv14_1, w=get_weight('conv14_2',(512,256,3,3)), b=get_bias('conv14_2',(512,)),quantize=True, d_scale_prev= 23.6808060224, w_scale=135.036341695, kernel=(3,3), stride=(2, 2))
# conv15_1    =convolution(input=conv14_2, w=get_weight('conv15_1',(128,512,1,1)), b=get_bias('conv15_1',(128,)),quantize=True, d_scale_prev= 26.9081833296, w_scale=363.399615913, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv15_2    =convolution(input=conv15_1, w=get_weight('conv15_2',(256,128,3,3)), b=get_bias('conv15_2',(256,)),quantize=True, d_scale_prev= 32.6812935783, w_scale=157.428430349, kernel=(3,3), stride=(2, 2))
# conv16_1    =convolution(input=conv15_2, w=get_weight('conv16_1',(128,256,1,1)), b=get_bias('conv16_1',(128,)),quantize=True, d_scale_prev= 20.7481958129, w_scale=283.463748659, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv16_2    =convolution(input=conv16_1, w=get_weight('conv16_2',(256,128,3,3)), b=get_bias('conv16_2',(256,)),quantize=True, d_scale_prev= 39.1085647054, w_scale=140.088296847, kernel=(3,3), stride=(2, 2))
# conv17_1    =convolution(input=conv16_2, w=get_weight('conv17_1',(64,256,1,1)), b=get_bias('conv17_1',(64,)),quantize=True, d_scale_prev= 25.8636239906, w_scale=181.295474825, kernel=(1,1), stride=(1, 1), pad=(0,0))
# conv17_2    =convolution(input=conv17_1, w=get_weight('conv17_2',(128,64,3,3)), b=get_bias('conv17_2',(128,)),quantize=True, d_scale_prev= 21.368959569, w_scale=98.4925936862, kernel=(3,3), stride=(2, 2))
#
#
# conv11_mbox_loc    =convolution(input=conv11, w=get_weight('conv11_mbox_loc',(12,512,1,1)), b=get_bias('conv11_mbox_loc',(12,)),quantize=True, d_scale_prev= 16.7585329697, w_scale=78.3555422285, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv11_mbox_conf    =convolution(input=conv11, w=get_weight('conv11_mbox_conf',(63,512,1,1)), b=get_bias('conv11_mbox_conf',(63,)),quantize=True, d_scale_prev= 16.7585329697, w_scale=23.8404454112, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv13_mbox_loc    =convolution(input=conv13, w=get_weight('conv13_mbox_loc',(24,1024,1,1)), b=get_bias('conv13_mbox_loc',(24,)),quantize=True, d_scale_prev= 5.46291092014, w_scale=137.15521692, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv13_mbox_conf    =convolution(input=conv13, w=get_weight('conv13_mbox_conf',(126,1024,1,1)), b=get_bias('conv13_mbox_conf',(126,)),quantize=True, d_scale_prev= 5.46291092014, w_scale=20.8723271414, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv14_2_mbox_loc    =convolution(input=conv14_2, w=get_weight('conv14_2_mbox_loc',(24,512,1,1)), b=get_bias('conv14_2_mbox_loc',(24,)),quantize=True, d_scale_prev= 26.9081833296, w_scale=148.255007027, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv14_2_mbox_conf    =convolution(input=conv14_2, w=get_weight('conv14_2_mbox_conf',(126,512,1,1)), b=get_bias('conv14_2_mbox_conf',(126,)),quantize=True, d_scale_prev= 26.9081833296, w_scale=21.0791147728, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv15_2_mbox_loc    =convolution(input=conv15_2, w=get_weight('conv15_2_mbox_loc',(24,256,1,1)), b=get_bias('conv15_2_mbox_loc',(24,)),quantize=True, d_scale_prev= 20.7481958129, w_scale=156.434490254, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv15_2_mbox_conf    =convolution(input=conv15_2, w=get_weight('conv15_2_mbox_conf',(126,256,1,1)), b=get_bias('conv15_2_mbox_conf',(126,)),quantize=True, d_scale_prev= 20.7481958129, w_scale=22.8982352066, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv16_2_mbox_loc    =convolution(input=conv16_2, w=get_weight('conv16_2_mbox_loc',(24,256,1,1)), b=get_bias('conv16_2_mbox_loc',(24,)),quantize=True, d_scale_prev= 25.8636239906, w_scale=163.664553431, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv16_2_mbox_conf    =convolution(input=conv16_2, w=get_weight('conv16_2_mbox_conf',(126,256,1,1)), b=get_bias('conv16_2_mbox_conf',(126,)),quantize=True, d_scale_prev= 25.8636239906, w_scale=28.246942679, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv17_2_mbox_loc    =convolution(input=conv17_2, w=get_weight('conv17_2_mbox_loc',(24,128,1,1)), b=get_bias('conv17_2_mbox_loc',(24,)),quantize=True, d_scale_prev= 14.1117994016, w_scale=262.461097438, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
# conv17_2_mbox_conf    =convolution(input=conv17_2, w=get_weight('conv17_2_mbox_conf',(126,128,1,1)), b=get_bias('conv17_2_mbox_conf',(126,)),quantize=True, d_scale_prev= 14.1117994016, w_scale=41.0664206634, kernel=(1,1), stride=(1, 1), pad=(0,0),relu=False)
#
# mbox_conf=get_mbox_conf(conv11_mbox_conf,conv13_mbox_conf,conv14_2_mbox_conf,conv15_2_mbox_conf,conv16_2_mbox_conf,conv17_2_mbox_conf)
# mbox_loc=get_mbox_loc(conv11_mbox_loc,conv13_mbox_loc,conv14_2_mbox_loc,conv15_2_mbox_loc,conv16_2_mbox_loc,conv17_2_mbox_loc)
# mbox_prior=get_mbox_prior(conv11,conv13,conv14_2,conv15_2,conv16_2,conv17_2)
#
# final_output = get_detection_out_tvm(mbox_conf, mbox_loc, mbox_prior, 1, num_anchor, 21)
