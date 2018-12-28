import caffe
import mxnet as mx
import numpy as np

CAFFE_NET = 'deploy.prototxt'
CAFFE_MODEL = 'mobilenet_iter_73000.caffemodel' #training model from Caffe with batchnorm layer

MXNET_MODEL = 'ssd_300_mobilenet1.0_voc_0000_0.1142.params' #only for reference
MXNET_SAVED='mobilenet_v1_300_rebuild_similar_parsed_step512_mboxloc-0000.params'


caffe_net = caffe.Net(CAFFE_NET, caffe.TEST)
caffe_net.copy_from(CAFFE_MODEL)
mbox=dict()
mbox['11']=0
mbox['13']=1
mbox['14_2']=2
mbox['15_2']=3
mbox['16_2']=4
mbox['17_2']=5
def get_params(cnet, mx_params):
    data_shape = cnet.blobs['data'].data.shape
    arg_params_map = mx_params.copy()
    #old version (wrong version)
    #aux_params_map = dict()
    expand=False
    for k, v in cnet.params.iteritems():
        conv='conv' in k
        dw ='dw' in k
        bn='bn' in k
        scale='scale' in k
        mbox_='mbox' in k
        conf='conf' in k
        layer_id = k.replace('/dw', '').replace('/bn', '').replace('conv', '').replace('_mbox', '').replace('_loc', '').replace('_conf', '')
        if (conv==True and bn == False and scale==False): #conv or conv/dw
            if(mbox_==True):
                if(conf==True):
                    w_name='class_predictors.'+str(mbox[layer_id])+'.predictor.weight'
                    b_name='class_predictors.'+str(mbox[layer_id])+'.predictor.bias'
                else: #loc
                    w_name='box_predictors.'+str(mbox[layer_id])+'.predictor.weight'
                    b_name='box_predictors.'+str(mbox[layer_id])+'.predictor.bias'
                x = v[0].shape[0]
                y = v[0].shape[1]
                data_raw = np.zeros((x,y,3,3))
                for i in range(x):
                    for j in range(y):
                        data_raw[i][j][1][1]=v[0].data[i][j][0][0]

                arg_params_map[w_name] = mx.nd.array(data_raw)
                arg_params_map[b_name] = mx.nd.array(v[1].data)

            else:
                if (layer_id == '14_1'):
                    expand = True
                if (expand == False):
                    if (dw == True):
                        layer_name = 'conv' + str(2 * int(layer_id) - 1)
                    else:
                        layer_name = 'conv' + str(2 * int(layer_id))
                    arg_name = 'features.mobilenet0_' + layer_name + '_weight'
                else:
                    arg_name = 'features.expand_conv' + layer_id + '_weight'
                arg_params_map[arg_name] = mx.nd.array(v[0].data)


        elif (bn==True):

            if(expand==False):
                if (dw == True):
                    layer_name = str(2 * int(layer_id) - 1)
                else:
                    layer_name = str(2 * int(layer_id))
                aux_name = 'features.mobilenet0_batchnorm' + layer_name
            else:
                aux_name = 'features.expand_bn' + layer_id

            #old version (wrong version)
            # aux_params_map[aux_name+'_running_mean'] = mx.nd.array(v[0].data)
            # aux_params_map[aux_name+'_running_var'] = mx.nd.array(v[1].data)
            # aux_params_map[aux_name+'_gamma'] = mx.nd.array(cnet.params[k.replace('bn', 'scale')][0].data)
            # aux_params_map[aux_name+'_beta'] = mx.nd.array(cnet.params[k.replace('bn', 'scale')][1].data)

            #new version

            if(expand==False):
                arg_params_map[aux_name + '_running_mean'] = mx.nd.array(v[0].data)
                arg_params_map[aux_name + '_running_var'] = mx.nd.array(v[1].data)
            else:
                arg_params_map[aux_name + '_moving_mean'] = mx.nd.array(v[0].data)
                arg_params_map[aux_name + '_moving_var'] = mx.nd.array(v[1].data)
            arg_params_map[aux_name + '_gamma'] = mx.nd.array(cnet.params[k.replace('bn', 'scale')][0].data)
            arg_params_map[aux_name + '_beta'] = mx.nd.array(cnet.params[k.replace('bn', 'scale')][1].data)

        elif (scale==True): #scale
            assert k.replace('scale', 'bn') in cnet.params
        else:
            raise ValueError

    return arg_params_map


loaded = mx.ndarray.load(MXNET_MODEL)
arg_params_map = get_params(caffe_net, loaded)

save_dict = {('%s' % k) : v for k, v in arg_params_map.items()}
mx.nd.save(MXNET_SAVED, save_dict)

print ('Save network done!')
