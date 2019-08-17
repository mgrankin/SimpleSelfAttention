from fastai.torch_core import *
import torch.nn as nn
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial

#Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)



# Adapted from SelfAttention layer at https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
# Inspired by https://arxiv.org/pdf/1805.08318.pdf
class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1, sym=False):#, n_out:int):
        super().__init__()
           
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)      
       
        self.gamma = nn.Parameter(tensor([0.]))
        
        self.sym = sym
        self.n_in = n_in
        
    def forward(self,x):
        
        
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)
                
        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)
        
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
          
        o = self.gamma * o + x
        
          
        return o.view(*size).contiguous()        
 
# adapted from https://github.com/fastai/fastai/blob/master/examples/train_imagenette.py
# added self attention parameter
# changed per gpu bs for bs_rat

from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
from fastai.vision.models.xresnet import *
#from fastai.vision.models.xresnet2 import *
#from fastai.vision.models.presnet import *
#from xresnet import *
from functools import partial

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_data(size, woof, bs, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder()
            #.transform(([flip_lr(p=0.5)], []), size=size)
            .transform(get_transforms(), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

from fastai.vision.models.xresnet import ResBlock
from torchvision.models.resnet import BasicBlock, Bottleneck

def inject_ssa(model, nlayer=-4, nblock=-1, sym=0): # really works only for resnet 
    res_part = model
    
    if isinstance(model, nn.Sequential) and not isinstance(model, XResNet): # resnet from fastai
        res_part = model[0]
    
    if isinstance(res_part, (XResNet, nn.Sequential)): # xresnet and resnet from fastai
        res_layers = [layer 
                     for layer in res_part 
                         if isinstance(layer, nn.Sequential) 
                            and any(isinstance(x, (BasicBlock, ResBlock, Bottleneck)) for x in layer)]
    elif isinstance(res_part, ResNet): # resnet from pytorch
        res_layers = [res_part.layer1, res_part.layer2, res_part.layer3, res_part.layer4]

    res_layer = res_layers[nlayer]
    res_block = res_layer[nblock]
    
    if isinstance(res_block, Bottleneck): # resnet
        out_channels = res_block.conv3.out_channels
        res_block.bn3 = nn.Sequential(res_block.bn3, SimpleSelfAttention(out_channels,ks=1,sym=sym))
    if isinstance(res_block, BasicBlock): # resnet
        out_channels = res_block.conv2.out_channels
        res_block.bn2 = nn.Sequential(res_block.bn2, SimpleSelfAttention(out_channels,ks=1,sym=sym))
    elif isinstance(res_block, ResBlock): # xresnet
        out_channels = res_block.convs[-1][0].out_channels
        res_block.convs = nn.Sequential(res_block.convs, SimpleSelfAttention(out_channels,ks=1,sym=sym))
    
    model.cuda()
    return model

from radam import *
from optimizers import *

@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    woof: Param("Use imagewoof (otherwise imagenette)", int)=0,
    lr: Param("Learning rate", float)=1e-3,
    size: Param("Size (px: 128,192,224)", int)=128,
    alpha: Param("Alpha", float)=0.99,
    mom: Param("Momentum", float)=0.9,
    eps: Param("epsilon", float)=1e-6,
    epochs: Param("Number of epochs", int)=5,
    bs: Param("Batch size", int)=256,
    mixup: Param("Mixup", float)=0.,
    opt: Param("Optimizer (adam,rms,sgd,radam,novograd)", str)='radam',
    arch: Param("Architecture (xresnet34, xresnet50, resnet50)", str)='xresnet50',
    pretrained: Param("Use pretrained weights", int)=0,
    sa: Param("Self-attention", int)=0,
    sym: Param("Symmetry for self-attention", int)=0,
    dump: Param("Print model; don't train", int)=0,
    lrfinder: Param("Run learning rate finder; don't train", int)=0,
    log: Param("Log file name", str)='log',
    ):
    "Distributed training of Imagenette."

    bs_one_gpu = bs
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='radam' : opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
    elif opt=='novograd' : opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)

    data = get_data(size, woof, bs)
    bs_rat = bs/bs_one_gpu   #originally bs/256
    if gpu is not None: bs_rat *= max(num_distrib(), 1)
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    lr *= bs_rat
    m = globals()[arch]
    log_cb = partial(CSVLogger,filename=log)

    learn = (cnn_learner(data, m, pretrained=pretrained,
                wd=1e-2, opt_func=opt_func,
                metrics=[accuracy,top_k_accuracy],
                bn_wd=False, true_wd=True,
                #loss_func = LabelSmoothingCrossEntropy(),
                callback_fns=[log_cb]))

    learn.unfreeze()

    if sa:
        inject_ssa(learn.model, sym=sym)

    print(learn.path)

    if dump: print(learn.model); exit()
    if mixup: learn = learn.mixup(alpha=mixup)
    learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    if lrfinder:
        # run learning rate finder
        IN_NOTEBOOK = 1
        learn.lr_find(wd=1e-2)
        learn.recorder.plot()
    else:
        learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
