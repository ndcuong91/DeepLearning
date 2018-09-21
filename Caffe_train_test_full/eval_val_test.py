import caffe
import cv2
import config
import os
import numpy as np


net_file= '/home/prdcv/Desktop/zaloAIchallenge/landmark/TrainVal/ResNet-152-deploy.prototxt'
caffe_model='/home/prdcv/Desktop/zaloAIchallenge/landmark/TrainVal/resnet_iter_180000.caffemodel'
mean_file='/home/prdcv/Desktop/zaloAIchallenge/landmark/TrainVal/mean.npy'

val_folder=config.data_folder+'/val'
test_folder='/home/prdcv/Desktop/zaloAIchallenge/landmark/Public'

shape= 224
channel=3  #3 for color image, 1 for gray


def init_caffe():
    caffe.set_mode_gpu()
    net = caffe.Net(net_file,caffe_model,caffe.TEST)
    net.blobs['data'].reshape(1,channel,shape,shape)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))  # if using RGB instead of BGR
    transformer.set_raw_scale('data', 255.0)
    return net, transformer

def classify_img(net,transformer, img_path):
    img = caffe.io.load_image(img_path)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    output = net.forward()
    return output


def val_accurary(net, transformer):
    classes = [os.path.join(val_folder, o) for o in os.listdir(val_folder) if
               os.path.isdir(os.path.join(val_folder, o))]
    for i in range(len(classes)):
        classes[i] = classes[i].replace(val_folder + '/', '')

    top3_result=0
    total_file=0
    classes=np.sort(classes)
    for i in range(len(classes)):
        if (classes[i] == 'train' or classes[i] == 'val' or classes[i] == 'train_lmdb' or classes[i] == 'val_lmdb'):
            continue
        val_class_folder = val_folder + '/' + classes[i]
        onlyfiles = [f for f in os.listdir(val_class_folder) if os.path.isfile(os.path.join(val_class_folder, f))]


        class_file=len(onlyfiles)
        good_classify=0
        for j in range(len(onlyfiles)):
            file_path=val_folder+'/'+classes[i]+'/'+onlyfiles[j]
            classify_img(net,transformer,file_path)
            best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1]

            for k in range(len(best_n)):
                if (str(best_n[k])==classes[i]):
                    good_classify+=1

        print("Class "+classes[i]+": " +str(good_classify)+"/"+str(class_file)+", accuracy: "+str(float(good_classify)/float(class_file)))
        total_file+=len(onlyfiles)
        top3_result+=good_classify

    print("Val accuracy: " +str(float(top3_result)/float(total_file)))

def eval_test(net, transformer):
    onlyfiles = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
    result='id,predicted\n'
    for j in range(len(onlyfiles)):
        result+=onlyfiles[j].replace('.jpg','')+','
        file_path = test_folder + '/' + onlyfiles[j]
        classify_img(net, transformer, file_path)
        best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1]

        for k in range(len(best_n)):
            if (k<2):
                result+=str(best_n[k])+' '
            else:
                result+=str(best_n[k])+'\n'

    with open(config.data_folder + '/submission_CuongND.csv', 'w') as file:
        file.write(result)

def main():

    network, transformer=init_caffe()
    val_accurary(network, transformer)
    #eval_test(network, transformer)

    print('end.')


if __name__ == '__main__':
    main()
