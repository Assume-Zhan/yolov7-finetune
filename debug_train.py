from copy import deepcopy
from pathlib import Path

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
from utils.general import labels_to_class_weights, init_seeds, one_cycle, fitness, increment_path
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

BATCH_SIZE = 8
EPOCHS = 20

def train():
    
    # Configuration
    batch_size = BATCH_SIZE
    device = select_device('0', batch_size = BATCH_SIZE)
    cuda = device.type != 'cpu'
    init_seeds(1)
    
    # Files
    with open("data/Aquarium/data.yaml") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Load data dict
    nc = int(data_dict['nc'])  # number of classes
    dataloader, dataset = create_dataloader(data_dict['train'], 640, batch_size, 32, hyp=hyp, augment=True)
 
    # Load Model and Dataset settings
    ckpt = torch.load('./weight/yolov7x_training.pt', map_location = device)
    model = Model("cfg/training/yolov7.yaml", ch = 3, nc = nc).to(device)
    state_dict = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict(), exclude=['anchor'])
    model.load_state_dict(state_dict, strict=False)
    
    # Optimizer, Goal : Get a larger batch size by accumulating gradient
    accumulate = max(round(64 / batch_size), 1)
    
    # Model parameters
    nl = model.model[-1].nl
    model.nc = nc
    model.names = data_dict['names']
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= 3. / nl
    # hyp["label_smoothing"] = 0.0
    # hyp["weight_decay"] *= BATCH_SIZE * accumulate / 64  # scale weight_decay

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

    # Parameter and Optimizer settings
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    lf = one_cycle(1, hyp['lrf'], EPOCHS)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']

    # Data and Weight Settings
    nb = len(dataloader)
    nw = max(round(hyp["warmup_epochs"] * nb), 1000)
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=640)
    model.half().float()
    scheduler.last_epoch = -1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)
    compute_loss = ComputeLoss(model)
    
    results = (0, 0, 0, 0, 0, 0, 0)
    
    del ckpt
    
    testloader, testset = create_dataloader(data_dict['val'], 640, batch_size * 2, 32, hyp=hyp, augment=True, pad = 0.5, rect = True)
    
    save_dir = Path(increment_path(Path("/yolov7/runs/train") / "exp", exist_ok=False))
    results_file = save_dir / "results.txt"
    weight_pt = save_dir / "weight.pt"
    
    # Start training
    for epoch in range(EPOCHS):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device = device)  # mean losses
        print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "labels", "img_size"))
        pbar = tqdm(enumerate(dataloader), total = nb)  # progress bar
        optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking = True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            
            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, 64 / BATCH_SIZE]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Forward
            with amp.autocast(enabled=cuda):
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
                    
            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
            s = ("%10s" * 2 + "%10.4g" * 6) % ("%g/%g" % (epoch, EPOCHS - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------
        scheduler.step()
        # end epoch ----------------------------------------------------------------------------------------------------
        
        # mAP
        ema.update_attr(model, include=["yaml", "nc", "hyp", "gr", "names", "stride", "class_weights"])
        final_epoch = epoch + 1 == EPOCHS
        results, maps, times = test.test(data_dict, batch_size=BATCH_SIZE * 2, imgsz=640, model=ema.ema, 
                                         dataloader=testloader, save_dir=save_dir, verbose=final_epoch, 
                                         plots=final_epoch, compute_loss=compute_loss)

        # Write
        with open(results_file, "a") as f:
            f.write(s + "%10.4g" * 7 % results + "\n")

        # Update best mAP
    # end training
    log = {"epoch": epoch, 
            "model": deepcopy(model).half(), 
            "ema": deepcopy(ema.ema).half(), 
            "updates": ema.updates, 
            "optimizer": optimizer.state_dict()}
    torch.save(log, weight_pt)
    del log
    
    torch.cuda.empty_cache()

if __name__ == '__main__':
    train()