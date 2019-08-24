# the code mostly from https://github.com/sdoria/SimpleSelfAttention

# adapted from https://github.com/fastai/fastai/blob/master/examples/train_imagenette.py
# added self attention parameter
# changed per gpu bs for bs_rat


from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
#from fastai.vision.models.xresnet import *
#from fastai.vision.models.xresnet2 import *
#from fastai.vision.models.presnet import *
from xresnet import *
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
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

from radam import *
from optimizers import *
from ranger import *
from ralamb import *
from over9000 import *

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
        opt: Param("Optimizer (adam,rms,sgd)", str)='adam',
        arch: Param("Architecture (xresnet34, xresnet50)", str)='xresnet50',
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
    elif opt=='ranger'  : opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    elif opt=='ralamb'  : opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
    elif opt=='over9000'  : opt_func = partial(Over9000,  betas=(mom,alpha), eps=eps)

    data = get_data(size, woof, bs)
    bs_rat = bs/bs_one_gpu   #originally bs/256
    if gpu is not None: bs_rat *= max(num_distrib(), 1)
    if not gpu: print(f'lr: {lr}; eff_lr: {lr*bs_rat}; size: {size}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    lr *= bs_rat

    m = globals()[arch]
    
    log_cb = partial(CSVLogger,filename=log)
    
    learn = (Learner(data, m(c_out=10, sa=sa, sym=sym), wd=1e-2, opt_func=opt_func,
             metrics=[accuracy,top_k_accuracy],
             bn_wd=False, true_wd=True,
             loss_func = LabelSmoothingCrossEntropy(),
             callback_fns=[log_cb])
            )
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
