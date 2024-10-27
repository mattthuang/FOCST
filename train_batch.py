import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from utils import get_lr, AvgMeter

import config as CFG
from dataset import *
from models import *
from torch.utils.data import DataLoader


import scanpy as sc
import argparse
from huggingface_hub import login

parser = argparse.ArgumentParser(description='DDP for CLIP')

parser.add_argument('--exp_name', type=str, default='clip', help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--max_epochs', type=int, default=100, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
# parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')

parser.add_argument('--world_size', default=3, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--model', type=str, default='auto', help='')

parser.add_argument('--gpu', type=str, help='gpus', default='0,1,2,3')
parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
parser.add_argument('--data', type=str, default='her2st', help='dataset name:{"her2st","cscc"}.')
parser.add_argument('--logger', type=str, default='../logs/my_logs', help='logger path.')
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN .')
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.') # Hist2STold = 4


def setup(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def collate_fn(batch):
    batch_size = 140  # Define your batch size here
    image_dict = {}
    
    # Group patches by image ID
    # print("Batch received with length:", len(batch))
    for item in batch:
        img, img_id, x, y, exp = item
        if img_id not in image_dict:
            image_dict[img_id] = []
        image_dict[img_id].append((img, x, y, exp))
    
    # print("Grouped patches by image ID:")
    # for img_id in image_dict:
    #     print(f"Image ID: {img_id}, Number of patches: {len(image_dict[img_id])}")
    
    # Create batches ensuring each batch contains patches from only one image
    batches = []
    for img_id, patches in image_dict.items():
        imgs, xs, ys, exp = zip(*patches)
        num_patches = len(imgs)
        if num_patches < 6:
            #print(f"Skipping image ID: {img_id} because it has only {num_patches} patches.")
            continue
        # print(f"Creating batches for image ID: {img_id} with {len(imgs)} patches.")
        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i + batch_size]
            batch_xs = xs[i:i + batch_size]
            batch_ys = ys[i:i + batch_size]
            batch_exp = exp[i:i + batch_size]
            imgs_array = np.stack(batch_imgs)
            imgs_tensor = torch.tensor(imgs_array)
            xs_tensor = torch.tensor(batch_xs)
            ys_tensor = torch.tensor(batch_ys)
            exp_array = np.stack(batch_exp)
            exp_tensor = torch.tensor(exp_array)
            batches.append((imgs_tensor, img_id, xs_tensor, ys_tensor, exp_tensor))
            # print(f"Batch created for image ID: {img_id} with {len(batch_imgs)} patches.")
    
    return batches


def build_loader(args, train_dir, test_dir):
    train = PatchDataset(train_dir)
    test = PatchDataset(test_dir)
    
    #train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank)
    #test_sampler = DistributedSampler(test, num_replicas=world_size, rank=rank)
    
    # train_loader = DataLoader(train, batch_size=512, sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
    # test_loader = DataLoader(test, batch_size=512, sampler=test_sampler, collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train, batch_size=140, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test, batch_size=140,  collate_fn=collate_fn, num_workers=0)
    
    return train_loader, test_loader

# train = "/data/lab_ph/matt/SGCL2ST/data/filtered_her2/train"
# test = "/data/lab_ph/matt/SGCL2ST/data/filtered_her2/test"
# train_loader, test_loader = build_loader(train, test)
# for batch in train_loader:
#     for imgs, img_id, xs, ys, exp in batch:
#         # Verify that all patches in this batch come from the same image
#         assert all(id == img_id for id in [img_id]), "Batch contains patches from different images!"
#         #print(f"Processing batch for image ID: {img_id} with {imgs.size(0)} patches.")
#         # Process each batch which contains patches from the same image
#         print(f'{exp.shape} expression count shape')
#         print(f'{xs} x COORD')
#         print(f'{ys} Y COORD')
#         print(f'{imgs.shape} iMGE SHAPE')

def train_epoch(model, epoch, train_loader, optimizer, args, rank, world_size, lr_scheduler=None):
    loss_meter = AvgMeter()
    contrastive_loss_meter = AvgMeter()
    rmse_loss_meter = AvgMeter()
    zinb_loss_meter = AvgMeter()
    #train_loader.sampler.set_epoch(epoch) 
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    #scaler = torch.cuda.amp.GradScaler()

    for batch in tqdm_object:
        # Get data from batch (from your original code)
        for imgs, img_id, xs, ys, exp in batch:

            coords = np.column_stack((xs, ys))
            print(coords.shape, "coordinate shape")
            adj = calcADJ(coords,args.neighbor)
            # adj = adj.squeeze(0)
            adj = torch.tensor(adj,dtype=torch.int64)
            oris = exp.squeeze(0)
            n_counts=exp.sum(1)
            sfs = n_counts / np.median(n_counts)
            exp = torch.tensor(exp.squeeze(0),dtype=torch.float32) 
            print(exp.shape, "exp shape in side the main script ")
            # print(f'{exp.dtype}, EXPRESSION TYPE')
            # print(F'{adj.dtype}, ADJ D TYPE')
            # loss = model(patch, center, exp, adj, oris, sfs)
            # with torch.cuda.amp.autocast():
            
            loss, contrastive_loss, rmse_loss, zinb_loss = model(imgs, exp, adj, oris, sfs, world_size)
            #loss, contrastive_loss, rmse_loss = model(patch, center, exp, adj, oris, sfs)
            
            optimizer.zero_grad()
            loss.backward()
            # scaler.scale(loss).backward()

            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"{name} grad is None")
            
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM) 
                param.grad.data /= dist.get_world_size()

            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()
            
            # print(xs.shape, "X SHAPE")
            count = len(xs)
            loss_meter.update(loss.item(), count)
            contrastive_loss_meter.update(contrastive_loss.item(), count)
            rmse_loss_meter.update(rmse_loss.item(), count)
            zinb_loss_meter.update(zinb_loss.item(), count)

            # tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
            tqdm_object.set_postfix(train_loss=loss_meter.avg, contra_loss=contrastive_loss_meter.avg, 
                                    rmse_loss=rmse_loss_meter.avg, zinb_loss=zinb_loss_meter.avg , lr=get_lr(optimizer))
            #tqdm_object.set_postfix(train_loss=loss_meter.avg, contra_loss=contrastive_loss_meter.avg, 
                                    #rmse_loss=rmse_loss_meter.avg , lr=get_lr(optimizer))
    

    return loss_meter


def test_epoch(args, model, test_loader, rank, world_size):
    loss_meter = AvgMeter()
    contrastive_loss_meter = AvgMeter()
    rmse_loss_meter = AvgMeter()
    zinb_loss_meter = AvgMeter()


    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        for imgs, img_id, xs, ys, exp in batch:
            
            coords = np.column_stack((xs, ys))

            adj = calcADJ(coords,args.neighbor)
            # adj = adj.squeeze(0)
            adj = torch.tensor(adj,dtype=torch.int64)
            oris = exp.squeeze(0)
            n_counts=exp.sum(1)
            sfs = n_counts / np.median(n_counts)
            exp = torch.tensor(exp.squeeze(0),dtype=torch.float32) 

            loss, contrastive_loss, rmse_loss, zinb_loss = model(imgs, exp, adj, oris, sfs, world_size)
            #loss, contrastive_loss, rmse_loss = model(patch, center, exp, adj, oris, sfs)


            count = len(xs)
            loss_meter.update(loss.item(), count)
            contrastive_loss_meter.update(contrastive_loss.item(), count)
            rmse_loss_meter.update(rmse_loss.item(), count)
            zinb_loss_meter.update(zinb_loss.item(), count)

            # tqdm_object.set_postfix(valid_loss=loss_meter.avg)
            tqdm_object.set_postfix(valid_loss=loss_meter.avg, contra_loss=contrastive_loss_meter.avg, 
                                    rmse_loss=rmse_loss_meter.avg, zinb_loss=zinb_loss_meter.avg)
            # tqdm_object.set_postfix(valid_loss=loss_meter.avg, contra_loss=contrastive_loss_meter.avg, 
            #                 rmse_loss=rmse_loss_meter.avg)

    return loss_meter


def main(rank, world_size):

    print("Logging into Huggingface...")
    login("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")
    print("Starting...")

    args = parser.parse_args()

    print('From Rank: {}, ==> Making model..'.format(rank))
    setup(rank, world_size)
    print('From Rank: {}, ==> Loading Data..'.format(rank))

    #load the data
    print('From Rank: {}, ==> Preparing data..'.format(rank))
    train = "/data/lab_ph/matt/SGCL2ST/data/filtered_her2/train"
    test = "/data/lab_ph/matt/SGCL2ST/data/filtered_her2/test"
    train_loader, test_loader = build_loader(args, train, test)
   
    model = myModel().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    #load the data
    # print('From Rank: {}, ==> Preparing data..'.format(rank))
    
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CFG.step, gamma=CFG.factor)


    # Train the model for a fixed number of epochs
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.max_epochs):
        print(f"Epoch: {epoch + 1}")
        # step = "epoch"
        # Train the model
        model.train()
        # train_loader.sampler.set_epoch(epoch) 
        # test_loader.sampler.set_epoch(epoch)
        train_loss = train_epoch(model, epoch, train_loader, optimizer, args,rank, world_size, lr_scheduler=None)


        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(args, model, test_loader, rank, world_size)
        
        # if rank == 0:
        #     lr_scheduler.step(test_loss.avg)
        
        #     if test_loss.avg < best_loss:
        #         if not os.path.exists(str(args.exp_name)):
        #             os.mkdir(str(args.exp_name))
        #         best_loss = test_loss.avg
        #         best_epoch = epoch

        #         torch.save(model.state_dict(), str(args.exp_name) + "/SGCL2ST_UNI_512.pt")
        #         print("Saved Best Model! Loss: {}".format(best_loss))

        # UNDER HERE IS OLD BEFORE DOING DISTRIBUTED SAMPLER
        # Update learning rate
        # lr_scheduler.step()
        lr_scheduler.step(test_loss.avg)
        
        # if test_loss.avg < best_loss and rank == 0:
        if test_loss.avg < best_loss and rank == 0:
            if not os.path.exists(str(args.exp_name)):
                os.mkdir(str(args.exp_name))
            best_loss = test_loss.avg
            best_epoch = epoch

            torch.save(model.state_dict(), str(args.exp_name) + "/SGCL2ST_UNI_brst_140.pt")
            print("Saved Best Model! Loss: {}".format(best_loss))

    print("Done!, final loss: {}".format(best_loss))
    print("Best epoch: {}".format(best_epoch))
    cleanup()

if __name__ == '__main__':
    world_size = 3    
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size
    )

