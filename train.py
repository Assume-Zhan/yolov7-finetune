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
from utils.general import labels_to_class_weights, init_seeds, one_cycle, increment_path, fitness
from utils.loss import ComputeLossOTA, ComputeLoss
from utils.torch_utils import ModelEMA, intersect_dicts

BATCH_SIZE = 8
EPOCHS = 55

def train():
    
    batch_size = BATCH_SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(1)
    
    # Open Required Files : data, cfg, hyperparameters
    with open("data/Aquarium/data.yaml") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        
    # Prepare data
    dataloader, dataset = create_dataloader(data_dict['train'], 640, batch_size, 32, hyp=hyp, augment=True)

    ckpt = torch.load('./weight/yolov7x_training.pt', map_location=device)
    nc = int(data_dict['nc'])
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=nc).to(device)
    state_dict = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict(), exclude=['anchor'])
    model.load_state_dict(state_dict, strict=False)
    print("Transferred %g/%g items" % (len(state_dict), len(model.state_dict())))  # report
    
    # Freeze
    freeze = [f"model.{x}." for x in range(0)]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False
    
    nl = model.model[-1].nl
    model.nc = nc  # attach number of classes to model
    model.names = data_dict["names"]
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (640 / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp["label_smoothing"] = 0.0
 
    # Optimizer, Goal : Get a larger batch size by accumulating gradient
    accumulate = max(round(64 / batch_size), 1)
    hyp["weight_decay"] *= BATCH_SIZE * accumulate / 64  # scale weight_decay

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
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2
    
    # Directories
    save_dir = Path(increment_path(Path("/yolov7/runs/train") / "exp", exist_ok=False))
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    weight_pt = wdir / "weight.pt"
    results_file = save_dir / "results.txt"

    # Init
    best_fitness = 0.0
    init_seeds(1)
    lf = one_cycle(1, hyp['lrf'], EPOCHS)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model)
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']
        
    # Results
    if ckpt.get("training_results") is not None:
        results_file.write_text(ckpt["training_results"])  # write results.txt
    
    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    
    nb = len(dataloader)  # number of batches
    
    # Anchors
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=640)
    model.half().float()  # pre-reduce anchor precision

    # Model parameters
    nw = max(round(hyp["warmup_epochs"] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = -1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)
    compute_loss = ComputeLoss(model)  # init loss class
    
    del ckpt
    
    testloader, testset = create_dataloader(data_dict['val'], 640, batch_size * 2, 32, hyp=hyp, augment=True, pad = 0.5, rect = True)
    
    # Start training
    for epoch in range(EPOCHS):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses
        print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "box", "obj", "cls", "total", "labels", "img_size"))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        optimizer.zero_grad()

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, 64 / BATCH_SIZE]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Forward
            with amp.autocast():
                pred = model(imgs)  # forward
                if "loss_ota" not in hyp or hyp["loss_ota"] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

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
        
        # mAP
        ema.update_attr(model, include=["yaml", "nc", "hyp", "gr", "names", "stride", "class_weights"])
        final_epoch = epoch + 1 == EPOCHS
        results, maps, times = test.test(data_dict, batch_size=BATCH_SIZE * 2, imgsz=640, model=ema.ema, 
                                         dataloader=testloader, save_dir=save_dir, verbose=final_epoch, 
                                         plots=final_epoch, compute_loss=compute_loss)

        # Write
        with open(results_file, "a") as f:
            f.write(s + "%10.4g" * 7 % results + "\n")  # append metrics, val_loss

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
            log = {"epoch": epoch, "best_fitness": best_fitness, "training_results": results_file.read_text(), 
                   "model": deepcopy(model).half(), "ema": deepcopy(ema.ema).half(), "updates": ema.updates, 
                   "optimizer": optimizer.state_dict(), "wandb_id": None}
            torch.save(log, weight_pt)
            del log

    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    train()