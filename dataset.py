import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class CLIP_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=1,flatten=True,ori=False,adj=False,prune='Grid',neighs=4): 
        super(CLIP_HER2ST, self).__init__()
        
        self.cnt_dir = 'SGCL2ST/data/her2st/data/ST-cnts'
        self.img_dir = 'SGCL2ST/data/her2st/data/ST-imgs'
        self.pos_dir = 'SGCL2ST/data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'SGCL2ST/data/her2st/data/ST-pat/lbl'
        self.r = 112//r

        #sgene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('/data/lab_ph/matt/SGCL2ST/data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        
        samples = names #[0:35]
        
        fold=[0,6,12,18,24,27,31,33]
        
        te_names = [samples[i] for i in fold]
        print("Test set names:", te_names)
        tr_names = list(set(samples)-set(te_names))
        print("Train set names:",tr_names)

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        # print('Loading imgs...')
        # self.img_dict = {}
        # for i in self.names:
        #     img_tensor = torch.Tensor(np.array(self.get_img(i)))
        #     self.img_dict[i] = img_tensor
        #     print(f"Image {i} shape: {img_tensor.shape}")

        
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            for name in self.names:
                idx = self.meta_dict[name].index
                lbl = self.lbl_dict[name]
                lbl = lbl.loc[idx,:]['label'].values
                self.label[name] = lbl
            
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        
        # self.exp_dict = {
        #     i:scp.transform.log(scp.normalize.library_size_normalize((m[self.gene_set].values))) 
        #     for i,m in self.meta_dict.items()
        # }

        '''
        # no normalization 
        # '''
        self.exp_dict = {
            i:(m[self.gene_set].values)
            for i,m in self.meta_dict.items()
        }

        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        #self.loc_dict = {i:m[['pixel_x','pixel_y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs) # i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
        exps = torch.Tensor(exps)

        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches

        data=[ID, im, positions, exps, centers]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        #data.append(torch.Tensor(centers))
        return data
    
    
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name +'.jpg'
        #fig_name = os.listdir(pre)[0]
        #path = pre+'/'+fig_name
        im = Image.open(pre)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
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

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df
    
# class CLIP_HER2ST(torch.utils.data.Dataset):
#     """Some Information about HER2ST"""
#     def __init__(self,train=True,fold=0,r=1,flatten=True,ori=False,adj=False,prune='Grid',neighs=4): 
#         super(CLIP_HER2ST, self).__init__()
        
#         self.cnt_dir = 'SGCL2ST/data/her2st/data/ST-cnts'
#         self.img_dir = 'SGCL2ST/data/her2st/data/ST-imgs'
#         self.pos_dir = 'SGCL2ST/data/her2st/data/ST-spotfiles'
#         self.lbl_dir = 'SGCL2ST/data/her2st/data/ST-pat/lbl'
#         # self.cnt_dir = 'data/her2st/data/ST-cnts'
#         # self.img_dir = 'data/her2st/data/ST_imgs'
#         # self.pos_dir = 'data/her2st/data/ST-spotfiles'
#         # self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
#         self.r = 112//r

#         gene_list = list(np.load('SGCL2ST/data/her_hvg_cut_1000.npy',allow_pickle=True))
#         #gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
#         self.gene_list = gene_list
#         names = os.listdir(self.cnt_dir)
#         names.sort()
#         names = [i[:2] for i in names]
#         self.train = train
#         self.ori = ori
#         self.adj = adj
        
#         samples = names #[0:35]
        
#         fold=[0,6,12,18,24,27,31,33]
        
#         te_names = [samples[i] for i in fold]
#         print("Test set names:", te_names)
#         tr_names = list(set(samples)-set(te_names))
#         print("Train set names:",tr_names)

#         if train:
#             self.names = tr_names
#         else:
#             self.names = te_names

#         print('Loading imgs...')
#         self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}

#         # print('Loading imgs...')
#         # self.img_dict = {}
#         # for i in self.names:
#         #     img_tensor = torch.Tensor(np.array(self.get_img(i)))
#         #     self.img_dict[i] = img_tensor
#         #     print(f"Image {i} shape: {img_tensor.shape}")

        
#         print('Loading metadata...')
#         self.meta_dict = {i:self.get_meta(i) for i in self.names}
#         self.label={i:None for i in self.names}
#         self.lbl2id={
#             'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
#             'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
#         }
#         if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
#             self.lbl_dict={i:self.get_lbl(i) for i in self.names}
#             for name in self.names:
#                 idx = self.meta_dict[name].index
#                 lbl = self.lbl_dict[name]
#                 lbl = lbl.loc[idx,:]['label'].values
#                 self.label[name] = lbl
            
#         elif train:
#             for i in self.names:
#                 idx=self.meta_dict[i].index
#                 if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
#                     lbl=self.get_lbl(i)
#                     lbl=lbl.loc[idx,:]['label'].values
#                     lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
#                     self.label[i]=lbl
#                 else:
#                     self.label[i]=torch.full((len(idx),),-1)
#         self.gene_set = list(gene_list)
        
#         # self.exp_dict = {
#         #     i:scp.transform.log(scp.normalize.library_size_normalize((m[self.gene_set].values))) 
#         #     for i,m in self.meta_dict.items()
#         # }

#         '''
#         # no normalization 
#         # '''
        
#         self.exp_dict = {
#             i:(m[self.gene_set].values)
#             for i,m in self.meta_dict.items()
#         }

#         if self.ori:
#             self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
#             self.counts_dict={}
#             for i,m in self.ori_dict.items():
#                 n_counts=m.sum(1)
#                 sf = n_counts / np.median(n_counts)
#                 self.counts_dict[i]=sf
#         self.center_dict = {
#             i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
#             for i,m in self.meta_dict.items()
#         }
#         self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
#         self.adj_dict = {
#             i:calcADJ(m,neighs) # i:calcADJ(m,neighs,pruneTag=prune)
#             for i,m in self.loc_dict.items()
#         }
#         self.patch_dict=dfd(lambda :None)
#         self.lengths = [len(i) for i in self.meta_dict.values()]
#         self.cumlen = np.cumsum(self.lengths)
#         self.id2name = dict(enumerate(self.names))
#         self.flatten=flatten
        
#     def __getitem__(self, index):
        
#         ID=self.id2name[index]
#         im = self.img_dict[ID]
            
#         # im = torch.Tensor(np.array(self.im))
#         exps = self.exp_dict[ID]
#         if self.ori:
#             oris = self.ori_dict[ID]
#             sfs = self.counts_dict[ID]
#         centers = self.center_dict[ID]
#         loc = self.loc_dict[ID]
#         adj = self.adj_dict[ID]
#         patches = self.patch_dict[ID]
#         positions = torch.LongTensor(loc)
#         patch_dim = 3 * self.r * self.r * 4
#         label=self.label[ID]
#         exps = torch.Tensor(exps)

#         # if patches is None:
#         #     n_patches = len(centers)
#         #     if self.flatten:
#         #         patches = torch.zeros((n_patches,patch_dim))
#         #     else:
#         #         patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
#         #     for i in range(n_patches):
#         #         center = centers[i]
#         #         x, y = center
#         #         patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
#         #         if self.flatten:
#         #             patches[i] = patch.flatten()
#         #         else:
#         #             patches[i]=patch.permute(2,0,1)
#         #     self.patch_dict[ID]=patches

#         data=[ID, im, positions, exps]
#         if self.adj:
#             data.append(adj)
#         if self.ori:
#             data+=[torch.Tensor(oris),torch.Tensor(sfs)]
#         data.append(torch.Tensor(centers))
#         return data
    
    
        
#     def __len__(self):
#         return len(self.exp_dict)

#     def get_img(self,name):
#         img_path = os.path.join(self.img_dir, name[0], name + '.jpg')
#         #fig_name = os.listdir(pre)[0]
#         #path = pre+'/'+fig_name
#         im = Image.open(img_path)
#         return im

#     def get_cnt(self,name):
#         path = self.cnt_dir+'/'+name+'.tsv'
#         df = pd.read_csv(path,sep='\t',index_col=0)

#         return df

#     def get_pos(self,name):
#         path = self.pos_dir+'/'+name+'_selection.tsv'
#         # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
#         df = pd.read_csv(path,sep='\t')

#         x = df['x'].values
#         y = df['y'].values
#         x = np.around(x).astype(int)
#         y = np.around(y).astype(int)
#         id = []
#         for i in range(len(x)):
#             id.append(str(x[i])+'x'+str(y[i])) 
#         df['id'] = id

#         return df

#     def get_meta(self,name,gene_list=None):
#         cnt = self.get_cnt(name)
#         pos = self.get_pos(name)
#         meta = cnt.join((pos.set_index('id')))

#         return meta

#     def get_lbl(self,name):
#         # path = self.pos_dir+'/'+name+'_selection.tsv'
#         path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
#         df = pd.read_csv(path,sep='\t')

#         x = df['x'].values
#         y = df['y'].values
#         x = np.around(x).astype(int)
#         y = np.around(y).astype(int)
#         id = []
#         for i in range(len(x)):
#             id.append(str(x[i])+'x'+str(y[i])) 
#         df['id'] = id
#         df.drop('pixel_x', inplace=True, axis=1)
#         df.drop('pixel_y', inplace=True, axis=1)
#         df.drop('x', inplace=True, axis=1)
#         df.drop('y', inplace=True, axis=1)
#         df.set_index('id',inplace=True)
#         return df
    
    
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, directory):

        self.directory = directory
        self.files = sorted([os.path.join(self.directory, f) for f in os.listdir(self.directory)])

        self.data = []
    
        # Extract metadata for lazy loading
        for file in self.files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            #her2
            img_id = parts[0]
            x  = int(parts[1])
            y = int(parts[2].split('.')[0])

            # ##cscc
            # #img_id = '_'.join(parts[0:3])
            # img_id = os.path.splitext(filename)[0]
            # #print(img_id, "IMAGE ID IN DATASET ")
            # x = int(parts[3])
            # y = int(parts[4].split('.')[0])  # Remove file extension

            # #brst
            # img_id = '_'.join(parts[0:2])
            # x = int(parts[2])
            # y = int(parts[3].split('.')[0])
            
            self.data.append((file, img_id, x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file, img_id, x, y = self.data[idx]
        with np.load(file) as data:
            image = data['image']
            exp = data['count']
        return image, img_id, x, y, exp
      
class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, predefined_batches):
        self.imgs_tensor, self.img_id, self.xs_tensor, self.ys_tensor, self.exp_tensor = predefined_batches

    def __len__(self):
        return len(self.imgs_tensor)

    def __getitem__(self):
        #print(idx, "IND#EX")
        #print(self.imgs_tensor.shape,"IMAGE SHAPE INSIDE DATASET" )
        # Extract the elements at the specified index across all tensors
        img_tensor = self.imgs_tensor
        #print(img_tensor.shape, "IMAGE TENSOR SHAPE AFTER INDEX")
        #img_id = self.img_id[idx]
        # print(self.xs_tensor.shape, "xstensor before indexing")
        # print(idx, "index")
        # print(self.xs_tensor[0], "x tensor at index 0")

        xs_tensor = self.xs_tensor
        #print(xs_tensor.shape, "X TESNOR SHAPE")
        # print(xs_tensor, "xs tensor inside the dataset")
        ys_tensor = self.ys_tensor
        exp_tensor = self.exp_tensor

        xs_cpu = self.xs_tensor.cpu().numpy()
        ys_cpu = self.ys_tensor.cpu().numpy()
        coords = np.column_stack((xs_cpu, ys_cpu))
        #print(len(coords), "COORDS LENGTH")
        #print(coords)
        adj = calcADJ(coords, 4)
        adj = torch.tensor(adj, dtype=torch.int64)
        oris = self.exp_tensor.squeeze(0)
        n_counts = self.exp_tensor.sum(1)
        n_counts = n_counts.cpu().numpy()
        sfs = n_counts / np.median(n_counts)
        
        return self.imgs_tensor, xs_tensor, ys_tensor, self.exp_tensor, adj, sfs, oris, self.img_id

class CLIP_SKIN(torch.utils.data.Dataset):
    """Some Information about CLIP_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=8):
        super(CLIP_SKIN, self).__init__()

        self.dir = 'SGCL2ST/data/GSE144240_RAW/'
        self.r = 112//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load('./SGCL2ST/data/skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        
        samples = names
        print("All sample names:", samples)

        fold=[0,3,6,9]
        
        te_names = [samples[i] for i in fold]
        print("Test set names:", te_names)
        tr_names = list(set(samples)-set(te_names))
        print("Train set names:",tr_names)

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        # if self.norm:
        #     self.exp_dict = {
        #         i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
        #         for i,m in self.meta_dict.items()
        #     }
        # else:
        #     self.exp_dict = {
        #         i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
        #         for i,m in self.meta_dict.items()
        #     }
        self.exp_dict = {
            i:(m[self.gene_set].values)
            for i,m in self.meta_dict.items()
        }
        
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs) # i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches

        
        #data=[ID, im, positions, exps, centers]
        data=[ID, patches, positions, exps, centers]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        #data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
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

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
    

class BRST(torch.utils.data.Dataset):

    def __init__(self, train=True, fold=0, r=4, flatten=False, ori=False, adj=False, prune='Grid', neighs=4): 
        super(BRST, self).__init__()
        
        self.cnt_dir = './SGCL2ST/data/brst/data/count_filtered'
        self.img_dir = './SGCL2ST/data/brst/data/image_stained'
        self.r = 112 // r

        # Load the gene list and filtered HVG indices
        gene_list = list(np.load('/data/lab_ph/matt/SGCL2ST/data/brst/data/count_filtered/filtered_gene_names.npy', allow_pickle=True))
        self.gene_list = gene_list
        
        self.filtered_indices = np.load('/data/lab_ph/matt/SGCL2ST/data/filtered_hvg_indices.npy')  # Load HVG indices
        self.gene_list_filtered = [self.gene_list[i] for i in self.filtered_indices]  # Get the 1000 HVG gene names

        self.subtype_dict = self.get_subtype(self.img_dir)

        names = []
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith('.jpg') and not file.endswith('_mask.jpg'):
                    name = os.path.splitext(file)[0]
                    names.append(name)

        names.sort()
        self.train = train
        self.ori = ori
        self.adj = adj

        samples = names
        fold = [0, 6, 12, 18, 24, 27, 31, 33]
        
        te_names = [samples[i] for i in fold]
        print("Test set names:", te_names)
        tr_names = list(set(samples) - set(te_names))
        print("Train set names:", tr_names)

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i, self.gene_list) for i in self.names}

        print('Loading imgs...')
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        self.gene_set = list(gene_list)
        self.exp_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}

        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf

        self.center_dict = {i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) for i, m in self.meta_dict.items()}
        self.loc_dict = {i: m[['pixel_x', 'pixel_y']].values for i, m in self.meta_dict.items()}
        self.adj_dict = {i: calcADJ(m, neighs) for i, m in self.loc_dict.items()}
        self.patch_dict = dfd(lambda: None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)

        # Get the full expression data
        exps = self.exp_dict[ID]
        
        # Filter the expression data based on the HVG indices
        exps_filtered = exps[:, self.filtered_indices]

        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]

        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        
        #exps = torch.Tensor(exps_filtered)  # Use the filtered gene expressions
        #exps = torch.Tensor(exps)
        
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)
            self.patch_dict[ID] = patches

        data = [ID, patches, positions, exps]

        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]

        return data

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        subtype = self.subtype_dict[name]
        patient = name[0:7]
        img_path = os.path.join(self.img_dir, subtype, patient, f"{name}.jpg")
        im = Image.open(img_path)
        return im

    def get_meta(self, name, gene_list):
        meta = []
        pixel_x = []
        pixel_y = []
        path = self.cnt_dir + '/' + name[0:7] + '/'

        for file in os.listdir(path):
            if file[:2] == name[8:10]:
                cnt_path = os.path.join(path, file)
                cnt = np.load(cnt_path)
                pixel_x.append(cnt["pixel"][0])
                pixel_y.append(cnt["pixel"][1])
                meta.append(cnt["count"])

        df = pd.DataFrame(meta, columns=gene_list)
        df["pixel_x"] = pixel_x
        df["pixel_y"] = pixel_y

        return df

    def get_subtype(self, img_dir):
        sub_dict = {}
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith('.jpg') and not file.endswith('_mask.jpg'):
                    name = os.path.splitext(file)[0]
                    subtype = os.path.basename(os.path.dirname(root))
                    sub_dict[name] = subtype
        return sub_dict
