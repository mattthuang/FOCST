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
from PIL import Image

cnt_dir = './SGCL2ST/data/her2st/data/ST-cnts'
img_dir = './SGCL2ST/data/her2st/data/ST-imgs'
pos_dir = './SGCL2ST/data/her2st/data/ST-spotfiles'
lbl_dir = './SGCL2ST/data/her2st/data/ST-pat/lbl' 
output_tr = 'SGCL2ST/data/filtered_her2/train'
output_te = 'SGCL2ST/data/filtered_her2/test'
gene_list = list(np.load('./SGCL2ST/data/her_hvg_cut_1000.npy',allow_pickle=True))

def get_ID():
    names = os.listdir(cnt_dir)
    names.sort()
    names = [i[:2] for i in names]
    samples = names
    fold=[0,6,12,18,24,27,31,33]
    te_names = [samples[i] for i in fold]
    print("Test set names:", te_names)
    tr_names = list(set(samples)-set(te_names))
    print("Train set names:",tr_names)
    return tr_names, te_names

def get_cnt(ID): 
    path = cnt_dir+'/'+ ID +'.tsv'
    df = pd.read_csv(path, sep='\t', index_col = 0)
    #df.rename( columns={'Unnamed: 0':'relative_coord'}, inplace=True )

    return df

def get_pos(name):
    path = pos_dir+'/'+name+'_selection.tsv'
    # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
    df = pd.read_csv(path,sep='\t')

    x = df['x'].values
    y = df['y'].values
    x = np.around(x).astype(int)
    y = np.around(y).astype(int)
    id = []
    for i in range(len(x)):
        id.append(str(x[i])+'x'+str(y[i])) 
    df['id'] = id

    return df

def get_meta(name):
    cnt = get_cnt(name)
    pos = get_pos(name)
    meta = cnt.join((pos.set_index('id')))
    meta.drop(["selected", "new_x", "new_y"], inplace = True, axis = 1)
    return meta

train, test = get_ID()
        
gene_columns = gene_list


for name in train:
    meta = get_meta(name)
    pre = img_dir+'/'+name[0]+'/'+name +'.jpg'
    im = np.array(Image.open(pre))
    print(f'Beign processing {name}')
    for index, row in meta.iterrows():
        # Extract the counts of the gene expression columns for the current row
        gene_counts = row[gene_columns].values
        
        x = int(row['pixel_x'])
        y = int(row['pixel_y'])
        
        # Extract the patch from the image
        #patch = im[(x - 112):(x + 112), (y - 112):(y + 112), :]
        #for patch size 518 for  giga-path 
        patch = im[(x - 112):(x + 112), (y - 112):(y    + 112), :]
        # Check if the patch has the correct dimensions
        if patch.shape == (224, 224, 3):
            
            # Define a unique filename for each row
            new_filename = os.path.join(output_tr, f'{name}_{x}_{y}.npz')
            
            # Save to .npz file 
            np.savez_compressed(new_filename, 
                                count=gene_counts, 
                                ID=name,                               
                                image=patch,
                                x=x, 
                                y=y
            )


    print(f'Finished processing {name}')


for name in test:
    meta = get_meta(name)
    pre = img_dir+'/'+name[0]+'/'+name +'.jpg'
    im = np.array(Image.open(pre))
    print(f'Beign processing {name}')
    for index, row in meta.iterrows():
        # Extract the counts of the gene expression columns for the current row
        gene_counts = row[gene_columns].values
        
        x = int(row['pixel_x'])
        y = int(row['pixel_y'])
        
        # Extract the patch from the image
        patch = im[(x - 112):(x + 112), (y - 112):(y + 112), :]
        
        # Check if the patch has the correct dimensions
        if patch.shape == (224, 224, 3):
            
            # Define a unique filename for each row
            new_filename = os.path.join(output_te, f'{name}_{x}_{y}.npz')
            
            # Save to .npz file 
            np.savez_compressed(new_filename, 
                                count=gene_counts, 
                                ID=name,                               
                                image=patch,
                                x=x, 
                                y=y
            )


    print(f'Finished processing {name}')