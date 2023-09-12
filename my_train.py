# Required Libraries
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from tqdm import tqdm

import test
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, check_img_size, set_logging, one_cycle, fitness
from utils.loss import ComputeLossOTA
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

# Setup hyper-parameters
class Settings():
    def __init__(self):
        self.stride = 32
        self.path = "/yolov7/data/Aquarium/train/images"
        self.image_size, self.imgsz_test = [check_img_size(x, self.stride) for x in [640, 640]]
        self.batch_size = 1
        self.anchor_t = 4

# train function
def train():
    weights = './weight/yolov7x_training.pt'
    device = 'cuda:0'
    epochs = 1
    
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
    # Load data
    settings = Settings()
    
    total_batch_size = settings.batch_size
    
    # Prepare Model
    number_of_class = 7
    name_of_class = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
    
    ckpt = torch.load(weights, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch = 3, nc = number_of_class, anchors = None).to(device)
    state_dict = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict(), exclude=['anchor'])
    model.load_state_dict(state_dict, strict=False)
    
    # Freeze
    freeze = [f"model.{x}." for x in range(0)]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False
    
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
    
    # EMA
    ema = ModelEMA(model)
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']
    del ckpt, state_dict
    
    dataloader, dataset = create_dataloader(settings.path, settings.image_size, settings.batch_size, max(int(model.stride.max()), 32), hyp=hyp, augment=True)
    
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    nb = len(dataloader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 1000)
    check_anchors(dataset, model=model, thr=settings.anchor_t, imgsz=settings.image_size)
    model.half().float()  # pre-reduce anchor precision
    
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= number_of_class / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (settings.image_size / 640) ** 2 * 3. / nl  # scale to image size and layers
    model.nc = number_of_class  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, number_of_class).to(device) * number_of_class  # attach class weights
    model.names = name_of_class
    
    scheduler.last_epoch = -1
    scaler = amp.GradScaler(enabled=True)
    compute_loss_ota = ComputeLossOTA(model)
    
    # Start training
    for epoch in range(epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(4, device=device)  # mean losses

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in tqdm(enumerate(dataloader), total=nb):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, 64 / settings.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Forward
            with amp.autocast(enabled=True):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # end batch ------------------------------------------------------------------------------------------------
        scheduler.step()
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    
    with open("data/AquariumDataset/data.yaml") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    testloader= create_dataloader("/yolov7/data/Aquarium/val/images", settings.image_size, settings.batch_size * 2, settings.stride, rect = True,
                                            pad = 0.5)

    results, maps, times = test.test(data_dict, 
                                     batch_size=settings.batch_size * 2, 
                                     imgsz = 640, 
                                     model=ema.ema, 
                                     dataloader=testloader, 
                                     save_dir='runs/train/prune', 
                                     verbose=True, 
                                     plots=True, 
                                     compute_loss=compute_loss_ota)
    
    
    results_file = 'runs/train/prune'
    mloss = (mloss * i + loss_items) / (i + 1)
    mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
    s = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
    with open(results_file, "a") as f:
        f.write(s + "%10.4g" * 7 % results + "\n")  # append metrics, val_loss

    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
    if fi > best_fitness:
        best_fitness = fi
        log = {"epoch": epoch, "best_fitness": best_fitness, "training_results": results_file.read_text(), "model": deepcopy(model).half(), "ema": deepcopy(ema.ema).half(), "updates": ema.updates, "optimizer": optimizer.state_dict(), "wandb_id": None}
        torch.save(log, 'runs/train/prune/weight.pt')
        del log
    
    # Save the last model
    ckpt = {
        'epoch': epochs,
        'model': deepcopy(model).half(),
        'ema': deepcopy(ema.ema).half(),
        'updates': ema.updates,
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, 'runs/train/prune/weight.pt')
    del ckpt
    
    torch.cuda.empty_cache()
    
    

# Main
if __name__ == "__main__":
        
    train()