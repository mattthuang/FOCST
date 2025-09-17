import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import torch
from torch import nn
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DeepSpeedPlugin
from graph_construction import calcADJ

import scanpy as sc
import argparse
from huggingface_hub import login

import config as CFG
from dataset import PatchDataset, CLIP_HER2ST, CLIP_SKIN, BRST, BatchDataset
from models import myModel
from utils import get_lr, AvgMeter


parser = argparse.ArgumentParser(description='Accelerate + DeepSpeed for CLIP')

parser.add_argument('--exp_name', type=str, default='clip', help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--max_epochs', type=int, default=100, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--model', type=str, default='auto', help='')
parser.add_argument('--gpu', type=str, help='gpus', default='0,1,2,3')
parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
parser.add_argument('--data', type=str, default='cscc', help='dataset name:{"her2st","cscc"}.')
parser.add_argument('--logger', type=str, default='../logs/my_logs', help='logger path.')
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN.')
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

# DeepSpeed-related config arguments if you want them in script
# (You can also omit these if you keep them all in accelerate config)
parser.add_argument('--deepspeed_config', type=str, default=None, help='Path to DeepSpeed config JSON if needed.')

def load_image_data(image_id, dataset):
    """
    Return everything you need for 'image_id':
      - patches: shape (N, 3, 224, 224)
      - gene_expr: shape (N, num_genes)
      - coords: shape (N, 2)
      etc.
    """

    ID, patches, exps, coords, _, ori, *_ = dataset[image_id]

    return ID, patches, exps, coords, ori

def pk_load(fold, mode='test',flatten=False,dataset='brst',r=4,ori=True,adj=True,prune='Grid',neighs=8): #r=4 Hist2ST
    assert dataset in ['her2st','cscc', 'brst']
    if dataset=='her2st':
        dataset = CLIP_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':
        dataset = CLIP_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )

    elif dataset == 'brst':
        dataset = BRST(train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    return dataset

def build_loaders_inference(args):
    print("Building loaders")
    trainset = pk_load(args.fold,'train',dataset=args.data,flatten=False,adj=True,ori=True,prune=args.prune)
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    testset = pk_load(args.fold,'test',dataset=args.data,flatten=False,adj=True,ori=True,prune=args.prune)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    print("Finished building loaders")
    return trainset, testset, train_loader, test_loader

#############################################################
# Training epoch
#############################################################
def train_epoch(model, trainset, optimizer, accelerator, train_ids, BatchDataset, args, lr_scheduler=None):
    """One epoch of training"""
    model.train()
    loss_meter = AvgMeter()
    contrastive_loss_meter = AvgMeter()
    rmse_loss_meter = AvgMeter()
    zinb_loss_meter = AvgMeter()

    # Wrap the loader with tqdm via the accelerator
    # This ensures only the main process logs the progress
    progress_bar = tqdm(train_ids, disable=not accelerator.is_local_main_process)

    for img_id in progress_bar: 
        ID, patches, exps, coords, ori = load_image_data(img_id, trainset)
        batch_ds = BatchDataset(ID, patches, exps, coords, ori)
        batch_dl = DataLoader(batch_ds, batch_size=170, shuffle=False)

        batch_dl = accelerator.prepare_data_loader(
            batch_dl, 
        )
        rank = accelerator.local_process_index
        print(
            f"[Rank={rank}] Starting image_id={img_id} with {len(batch_ds)} patches total."
        )

        total_patches_seen = 0
        for step, (IDs, img_patches, gene_expr, crds, oris) in enumerate(batch_dl):

            adj = calcADJ(crds.cpu().numpy(), args.neighbor)
            adj = torch.tensor(adj, dtype=torch.int64, device=accelerator.device)
            n_counts = gene_expr.sum(1)
            sfs = n_counts / np.median(n_counts.cpu().numpy())
            sfs = torch.tensor(sfs, dtype=torch.float32, device=accelerator.device)
            gene_expr = torch.tensor(gene_expr.squeeze(0), dtype=torch.float32)

            loss, contrastive_loss, rmse_loss, zinb_loss = model(img_patches, gene_expr, crds, adj, oris, sfs)
            
            # Backward
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            count = len(crds)
            loss_meter.update(loss.item(), count)
            contrastive_loss_meter.update(contrastive_loss.item(), count)
            rmse_loss_meter.update(rmse_loss.item(), count)
            zinb_loss_meter.update(zinb_loss.item(), count)

        progress_bar.set_postfix({
            "train_loss": loss_meter.avg,
            "contra_loss": contrastive_loss_meter.avg,
            "rmse_loss": rmse_loss_meter.avg,
            "zinb_loss": zinb_loss_meter.avg,
            "lr": get_lr(optimizer)
        })

    return loss_meter


#############################################################
# Validation epoch
#############################################################
def test_epoch(model, testset, accelerator, test_ids, BatchDataset, args):

    """One epoch of validation"""
    model.eval()
    loss_meter = AvgMeter()
    contrastive_loss_meter = AvgMeter()
    rmse_loss_meter = AvgMeter()
    zinb_loss_meter = AvgMeter()

    progress_bar = tqdm(test_ids, disable=not accelerator.is_local_main_process)

    with torch.no_grad():
     for img_id in progress_bar: 
        ID, patches, exps, coords, ori = load_image_data(img_id, testset)
        batch_ds = BatchDataset(ID, patches, exps, coords, ori)
        batch_dl = DataLoader(batch_ds, batch_size=170, shuffle=False)

        batch_dl = accelerator.prepare_data_loader(
            batch_dl, 
        )
        rank = accelerator.local_process_index
        print(
            f"[Rank={rank}] Starting image_id={img_id} with {len(batch_ds)} patches total."
        )

        total_patches_seen = 0
        for step, (IDs, img_patches, gene_expr, crds, oris) in enumerate(batch_dl):
            adj = calcADJ(crds.cpu().numpy(), args.neighbor)
            adj = torch.tensor(adj, dtype=torch.int64, device=accelerator.device)
            n_counts = gene_expr.sum(1)
            sfs = n_counts / np.median(n_counts.cpu().numpy())
            sfs = torch.tensor(sfs, dtype=torch.float32, device=accelerator.device)
            gene_expr = torch.tensor(gene_expr.squeeze(0), dtype=torch.float32)

            loss, contrastive_loss, rmse_loss, zinb_loss = model(img_patches, gene_expr, crds, adj, oris, sfs)
  

            count = len(crds)
            loss_meter.update(loss.item(), count)
            contrastive_loss_meter.update(contrastive_loss.item(), count)
            rmse_loss_meter.update(rmse_loss.item(), count)
            zinb_loss_meter.update(zinb_loss.item(), count)

        progress_bar.set_postfix({
            "train_loss": loss_meter.avg,
            "contra_loss": contrastive_loss_meter.avg,
            "rmse_loss": rmse_loss_meter.avg,
            "zinb_loss": zinb_loss_meter.avg,
        })

    return loss_meter

#############################################################
# Main
#############################################################
def main():
    # Parse arguments
    args = parser.parse_args()
    set_seed(args.seed)

    print("Logging into Huggingface...")
    login() #hugging face token here
    print("Starting...")

    # Create Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # set your preferences
        mixed_precision='no',  # or 'fp16', 'bf16'
        kwargs_handlers=[ddp_kwargs]
        # gradient_accumulation_steps=1, # example
    )

    #Prepare data
    # train_ids = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
    #              'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    #              'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    #             'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
    #             'F1', 'F2', 'F3', 'E1', 'E2', 'E3']
    
    # test_ids = ['G1', 'G2', 'G3', 'H1', 'H2', 'H3']
    

    train_ids = ['P5_ST_rep3', 'P5_ST_rep2', 'P5_ST_rep1',
                 'P9_ST_rep1', 'P9_ST_rep2',  'P9_ST_rep3',
                 'P2_ST_rep1','P2_ST_rep2', 'P2_ST_rep3'
                 ]
    
    test_ids = ['P10_ST_rep1', 'P10_ST_rep2', 'P10_ST_rep3']

    # train_ids = ['BC23269_C1', 'BC23269_C2', 'BC23269_D1', 'BC23272_D2', 'BC23272_E1', 
    #              'BC23272_E2', 'BC23277_D2', 'BC23277_E1', 'BC23277_E2', 'BC23288_D2', 
    #              'BC23288_E1', 'BC23288_E2', 'BC23377_C1', 'BC23377_C2', 'BC23377_D1', 
    #              'BC23450_D2', 'BC23450_E1', 'BC23450_E2', 'BC23506_C1', 'BC23506_C2', 
    #              'BC23506_D1', 'BC23508_D2', 'BC23508_E1', 'BC23508_E2', 'BC23803_D2', 
    #              'BC23803_E1', 'BC23803_E2', 'BC23810_D2', 'BC23810_E1', 'BC23810_E2', 
    #              'BC23895_C1', 'BC23895_C2', 'BC23895_D1', 'BC23901_C2', 'BC23901_D1', 
    #              'BC23903_C1', 'BC23903_C2', 'BC23903_D1', 'BC23944_D2', 'BC23944_E1', 
    #              'BC23944_E2', 'BC24044_D2', 'BC24044_E1', 'BC24044_E2', 'BC24105_C1', 
    #              'BC24105_C2', 'BC24105_D1', 'BC24220_D2', 'BC24220_E1', 'BC24220_E2', 
    #              'BC24223_D2', 'BC24223_E1', 'BC24223_E2']

    # test_ids = ['BC23287_C1', 'BC23287_C2', 'BC23287_D1', 'BC23567_D2', 'BC23567_E1', 
    #             'BC23567_E2', 'BC23268_C1', 'BC23268_C2', 'BC23268_D1', 'BC23270_D2', 
    #             'BC23270_E1', 'BC23270_E2', 'BC23209_C1', 'BC23209_C2', 'BC23209_D1']
    
    trainset, testset, *_ = build_loaders_inference(args)

    # Instantiate model
    model = myModel(accelerator)

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    # Prepare everything with Accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.max_epochs):
        accelerator.print(f"Epoch: {epoch + 1}/{args.max_epochs}")

        # Train
        train_loss_meter = train_epoch(model, trainset, optimizer, accelerator, train_ids, BatchDataset, args, lr_scheduler)
        train_loss_avg = train_loss_meter.avg

        # Validate
        test_loss_meter = test_epoch(model, testset, accelerator, test_ids, BatchDataset, args)
        test_loss_avg = test_loss_meter.avg

        # Step LR scheduler
        lr_scheduler.step(test_loss_avg)

        # Accelerator main process saves the best model
        if test_loss_avg < best_loss and accelerator.is_main_process:
            best_loss = test_loss_avg
            best_epoch = epoch
            if not os.path.exists(str(args.exp_name)):
                os.mkdir(str(args.exp_name))

            # Save model weights (Accelerator automatically gathers them)
            accelerator.save(
                model.state_dict(), 
                os.path.join(str(args.exp_name), "SGCL2ST_UNI_cscc_zeroshot_170.pt")
            )
            accelerator.print(f"Saved Best Model! Loss: {best_loss}")

        accelerator.print(
            f"[Epoch {epoch + 1}] train_loss: {train_loss_avg:.4f} | test_loss: {test_loss_avg:.4f}"
        )

    accelerator.print(f"Done! Best loss: {best_loss}, Best epoch: {best_epoch}")


if __name__ == '__main__':
    main()
