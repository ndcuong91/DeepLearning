import torch
import torchvision
import mxnet as mx


MXNET_MODEL = '/home/prdcv265/PycharmProjects/gvh205/gluon-cv/scripts/detection/ssd/final_model_for_pruning_28122018/ssd_300_mobilenet1.0_voc_best.params'

class Conv:
    weight=None
    bias=None
    in_channels=1
    out_channels=1
    kernel_size=None
    stride=None
    padding=None

class Batchnorm:
    weight=None
    bias=None
    running_var = None
    running_mean = None
    eps = 1e-5


def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_( b_conv + b_bn )
    #
    # we're done
    return fusedconv


torch.set_grad_enabled(False)
mxnet_params = mx.ndarray.load(MXNET_MODEL)

x = torch.randn(1, 64, 75, 75)

conv_torch = torch.nn.Conv2d(64,128,kernel_size=(1,1),stride=(1,1), padding=(0,0),bias=False)
mm=torch.tensor(mxnet_params['features.mobilenet0_conv4_weight'].asnumpy())
conv_torch.weight.copy_(torch.tensor(mxnet_params['features.mobilenet0_conv4_weight'].asnumpy()))
conv_torch.training=False

bn_torch=torch.nn.BatchNorm2d(128)
bn_torch.weight.copy_(torch.tensor(mxnet_params['features.mobilenet0_batchnorm4_gamma'].asnumpy()))
bn_torch.bias.copy_(torch.tensor(mxnet_params['features.mobilenet0_batchnorm4_beta'].asnumpy()))
bn_torch.running_mean=torch.tensor(mxnet_params['features.mobilenet0_batchnorm4_running_mean'].asnumpy())
bn_torch.running_var=torch.tensor(mxnet_params['features.mobilenet0_batchnorm4_running_var'].asnumpy())
bn_torch.training=False


net = torch.nn.Sequential(conv_torch,bn_torch)
y1 = net.forward(x)

fusedconv = fuse_conv_and_bn(conv_torch, bn_torch)
y2 = fusedconv.forward(x)
k=y1-y2
d = (y1 - y2).norm().div(y1.norm()).item()
print("error: %.8f" % d)
