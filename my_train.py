# Required Libraries
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from tqdm import tqdm

from models.yolo import Model

from utils.datasets import create_dataloader
from utils.general import check_img_size, one_cycle
from utils.torch_utils import intersect_dicts

# Setup hyper-parameters
class Settings():
    def __init__(self, setting_dict):
        self.path = setting_dict['path']
        self.image_size = setting_dict['image_size']
        self.single_cls = False
        self.batch_size = setting_dict['batch_size']

# Get dataset
def get_dataloader(settings, stride):
    
    # Return dataloader, dataset
    return create_dataloader(
        settings.path, 
        settings.image_size, 
        settings.batch_size, 
        stride, 
        settings
    )

# train function
def train(opt):
    weights = opt['weights']
    device = 'cuda:0'
    epochs = opt['epochs']
    
    # Load data
    stride = 32
    settings = {
        'path': "/yolov7/data/Aquarium/train/images",
        'image_size': [768, 1024],
        'batch_size': 1
    }
    settings['image_size'], imgsz_test = [check_img_size(x, stride) for x in settings['image_size']]
    settings = Settings(settings)
    
    total_batch_size = settings.batch_size
    
    dataloader, dataset = get_dataloader(settings, stride)
    
    # Prepare Model
    number_of_class = 7
    name_of_class = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    
    ckpt = torch.load(weights, map_location=device)
    model = Model(ckpt['model'].yaml, ch = 3, nc = number_of_class, anchors = None).to(device)
    
    exclude = ['anchor'] # Not understand what it does
    state_dict = ckpt['model'].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
    model.load_state_dict(state_dict, strict=False)
    
    # Prepare Optimizier
    weight_decay = 0.0005 # hyper
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    optimizer = optim.SGD(pg0, lr = 0.01, momentum = 0.937, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2
    
    # Prepare LR scheduler
    lf = one_cycle(1, 0.1, epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf)
    
    # Start training
    start_epoch = 0
    
    
    # Test
    

# Main
if __name__ == "__main__":
        
    opt = {
        'weights': './weight/yolov7x_training.pt',
        'epochs': 1
    }
        
    train(opt)