import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import config as CFG
from modules import *
import torch.distributed as dist


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
#         spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

class CLIPModel_resnet50(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=2048,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet50()
#         self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, patches, centers, exps, adj, aug=False):
        image_features = self.image_encoder(patches)
        spot_features = exps
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


# class myModel(nn.Module):
#     def __init__(
#         self,
#         temperature=CFG.temperature,
#         image_embedding=CFG.image_embedding,
#         spot_embedding=CFG.spot_embedding, #785,171
#         projection_dim = CFG.projection_dim, #256 
#     ):
#         super().__init__()
#         self.spot_embedding = spot_embedding
#         # self.image_encoder = ImageEncoder()
#         self.image_encoder = ImageEncoder_resnet152()
#         #self.image_encoder = ResNetForImageClassification.from_pretrained("./modules/microsoft/resnet-50/")
#         self.spot_encoder = SpotEncoder()
#         self.spot_autoencoder = SpotAutoEncoder(n_genes = spot_embedding)
#         self.image_projection = ProjectionHead(embedding_dim = image_embedding) #aka the input dim, 2048 for resnet50
#         self.spot_projection = ProjectionHead(embedding_dim = spot_embedding) #3467 shared hvgs  projection_dim spot_embedding
#         self.temperature = temperature
#         self.rmse = 0.1
#         self.zinb = 0.25


#     def forward(self, patches, centers, exps, adj, oris, sfs):
#         # Getting Image and spot Features
#         image_features = self.image_encoder(patches)
#         # spot_features = exps
#         spot_features = self.spot_encoder(exps, adj)
#         # print("image_features.shape = ", image_features.shape)
#         # print("spot_features.shape = ", spot_features.shape)
        
#         # Getting Image and Spot Embeddings (with same dimension) 
#         image_embeddings = self.image_projection(image_features)
#         spot_embeddings = self.spot_projection(spot_features)
#         # print("image_embeddings.shape = ", image_embeddings.shape)
#         # print("spot_embeddings.shape = ", spot_embeddings.shape)
        
#         spot_encoding = self.spot_autoencoder.encode(spot_embeddings, adj)
#         spot_reconstruction, extra = self.spot_autoencoder.decode(spot_encoding)
#         # print("spot_encoding.shape = ", spot_encoding.shape)
#         # print("spot_reconstruction.shape = ", spot_reconstruction.shape)
        
      
#         # Calculating the Contrastive Loss
#         logits = (spot_embeddings @ image_embeddings.T) / self.temperature
#         images_similarity = image_embeddings @ image_embeddings.T
#         spots_similarity = spot_embeddings @ spot_embeddings.T
#         targets = F.softmax(
#             (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
#         )
#         spots_loss = cross_entropy(logits, targets, reduction='none')
#         images_loss = cross_entropy(logits.T, targets.T, reduction='none')
#         contrastive_loss =  ((images_loss + spots_loss) / 2.0).mean() # shape: (batch_size)
        
#         #### Calculating the Reconstruction Loss
#         ## RMSE Loss
#         rmse_loss = torch.sqrt(nn.MSELoss()(spot_reconstruction, exps))
#         # ZINB Loss
#         m,d,p=extra
#         zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
#         ## Reconstruction_loss
#         reconstruction_loss = self.rmse * rmse_loss + self.zinb * zinb_loss
#         #reconstruction_loss = self.rmse * rmse_loss
        
#         return contrastive_loss + reconstruction_loss, contrastive_loss, rmse_loss, zinb_loss
#         #return contrastive_loss + reconstruction_loss, contrastive_loss, rmse_loss
'''
new stuff that i edited 
'''
class myModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding, #785,171
        projection_dim = CFG.projection_dim, #256 
    ):
        super().__init__()
        self.spot_embedding = spot_embedding
        # self.image_encoder = ImageEncoder()
        self.image_encoder = ImageEncoder_resnet152()
        #self.image_encoder = path_UNI()
        #self.image_encoder = CONCH()
        #self.image_encoder = path_giga()
        self.spot_encoder = SpotEncoder()
        self.spot_autoencoder = SpotAutoEncoder(n_genes = spot_embedding)
        self.image_projection = ProjectionHead(embedding_dim = image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim = spot_embedding) #3467 shared hvgs  projection_dim spot_embedding
        self.temperature = temperature
        self.rmse = 0.1
        self.zinb = 0.25

    def forward(self, image, exps, adj, oris, sfs):
        #image_features = []
        #n_patches = len(centers[0])
        # for i in range(n_patches):
        #     x, y = centers[0][i]
        #     x = x.item()
        #     y = y.item()
        #     patch = image[0, (x - 112):(x + 112), (y - 112):(y + 112), :]
        #     patch=patch.permute(2,0,1)
        #     patch_features = self.image_encoder(patch)  # Process one patch at a time
        #     #print("thisis PATCH _FEATURES", patch_features.shape)
        #     image_features.append(patch_features.detach())
        #     # print(len(image_features), "processed pathces")

        # image_features = torch.cat(image_features, dim=0)

        #uncomment below when training with batches
        #image=image.permute(0,3,1,2) 
        image = image.permute(2,0,1)
        image = image.float() / 255.0
        # print(image)
        print(image.shape, "image shape in models")
        image_features = self.image_encoder(image)
        #image_features = self.image_encoder(patch)

        exps = exps.to(torch.float32)
        print(adj.shape, "adj shape", adj.dtype)
        print(exps.shape, "exp shape", exps.dtype)
        print(sfs.shape, "sfs shape in model")

        spot_features = self.spot_encoder(exps, adj)

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)
        
        spot_encoding = self.spot_autoencoder.encode(spot_embeddings, adj)
        spot_reconstruction, extra = self.spot_autoencoder.decode(spot_encoding)
        
        # Calculating the Contrastive Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        contrastive_loss =  ((images_loss + spots_loss) / 2.0).mean() # shape: (batch_size)
        
        #### Calculating the Reconstruction Loss
        ## RMSE Loss
        #print(spot_reconstruction.shape, 'reconstruction shape')
        #print(exps.shape, "EXPRESSION SHAPE")

        rmse_loss = torch.sqrt(nn.MSELoss()(spot_reconstruction, exps))

        ## ZINB Loss
        m,d,p=extra

        zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
        ## Reconstruction_loss
        reconstruction_loss = self.rmse * rmse_loss + self.zinb * zinb_loss
        
        return contrastive_loss + reconstruction_loss, contrastive_loss, rmse_loss, zinb_loss

    # def forward(self, image, exps, adj, oris, sfs, world_size):
    #     rank = dist.get_rank()
    #     num_gpus = world_size
    # #def forward(self, image, centers, exps, adj, oris, sfs):
    #     # Getting Image and spot Features
    #     # image_features = []
    #     # n_patches = len(centers[0])
    #     # for i in range(n_patches):
    #     #     x, y = centers[0][i]
    #     #     x = x.item()
    #     #     y = y.item()
    #     #     patch = image[0, (x - 112):(x + 112), (y - 112):(y + 112), :]
    #     #     patch=patch.permute(2,0,1)
    #     #     patch_features = self.image_encoder(patch)  # Process one patch at a time
    #     #     #print("thisis PATCH _FEATURES", patch_features.shape)
    #     #     image_features.append(patch_features.detach())
    #     #     # print(len(image_features), "processed pathces")

    #     # image_features = torch.cat(image_features, dim=0)

    #     #uncomment below when training with batches
    #     image=image.permute(0,3,1,2) 
    #     #image = image.permute(2,0,1)
    #     image = image.float() / 255.0
    #     # print(image)

    #     print(image.shape, 'IMAGE SHAPE IN MODEL BEFORE IMAGE ENCODER')
    #     print(exps.shape, "EXPRESSION SHAPE IN MODEL BEFORE SPOT ENCODER")
    #     image_features = self.image_encoder(image)
    #     #image_features = self.image_encoder(patch)


    #     spot_features = self.spot_encoder(exps, adj)

    #     # image_embeddings = self.image_projection(image_features)
    #     # spot_embeddings = self.spot_projection(spot_features)

    #     image_features = image_features.contiguous()
        
    #     print("image_features.shape = ", image_features.shape)
    #     print("spot_features.shape = ", spot_features.shape)

    #     gathered_image_features = [torch.zeros_like(image_features) for _ in range(num_gpus)]
    #     gathered_spot_features = [torch.zeros_like(spot_features) for _ in range(num_gpus)]

    #     dist.all_gather(gathered_image_features, image_features)
    #     dist.all_gather(gathered_spot_features, spot_features)

    #     gathered_image_features[dist.get_rank()] = image_features
    #     gathered_spot_features[dist.get_rank()] = spot_features
        
    #     concatenated_image = torch.cat(gathered_image_features, dim=0)
    #     concatenated_spot = torch.cat(gathered_spot_features, dim=0)

    #     print(f"Rank {rank} - Concatenated image features shape: {concatenated_image.shape}")
    #     print(f"Rank {rank} - Concatenated spot features shape: {concatenated_spot.shape}")


    #     print(f"Rank {rank} - Concatenated image shape after broadcast: {concatenated_image.shape}")
    #     print(f"Rank {rank} - Concatenated spot shape after broadcast: {concatenated_spot.shape}")


    #     print(f"Rank {rank} - Shape before projection: {concatenated_image.shape}")
    #     image_embeddings = self.image_projection(concatenated_image)
    #     print(f"Rank {rank} - Image embeddings shape: {image_embeddings.shape}")

    #     spot_embeddings = self.spot_projection(concatenated_spot)
    #     print(f"Rank {rank} - Spot embeddings shape: {spot_embeddings.shape}")

    #     spot_encoding = self.spot_autoencoder.encode(spot_embeddings, adj)
    #     spot_reconstruction, extra = self.spot_autoencoder.decode(spot_encoding)
    #     print("spot_encoding.shape = ", spot_encoding.shape)
    #     print("spot_reconstruction.shape = ", spot_reconstruction.shape)
        
    #     # Calculating the Contrastive Loss
    #     logits = (spot_embeddings @ image_embeddings.T) / self.temperature
    #     images_similarity = image_embeddings @ image_embeddings.T
    #     spots_similarity = spot_embeddings @ spot_embeddings.T
    #     targets = F.softmax(
    #         (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
    #     )
    #     spots_loss = cross_entropy(logits, targets, reduction='none')
    #     images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    #     contrastive_loss =  ((images_loss + spots_loss) / 2.0).mean() # shape: (batch_size)
        
    #     #### Calculating the Reconstruction Loss
    #     ## RMSE Loss
    #     #print(spot_reconstruction.shape, 'reconstruction shape')
    #     #print(exps.shape, "EXPRESSION SHAPE")

    #     gathered_exps = [torch.zeros_like(exps) for _ in range(num_gpus)]
    #     dist.all_gather(gathered_exps, exps)
    #     gathered_exps_tensor = torch.cat(gathered_exps, dim = 0)

    #     print(gathered_exps_tensor.shape, "gathered expression tensor shpae")

    #     rmse_loss = torch.sqrt(nn.MSELoss()(spot_reconstruction, gathered_exps_tensor))
    #     # rmse_loss = torch.sqrt(nn.MSELoss()(spot_reconstruction, exps))

    #     ## ZINB Loss
    #     m,d,p=extra

    #     gathered_sfs = [torch.zeros_like(sfs) for _ in range(num_gpus)]
    #     dist.all_gather(gathered_sfs, sfs)
    #     gathered_sfs_tensor = torch.cat(gathered_sfs, dim = 0)
    #     #print(gathered_sfs_tensor.shape, "GATHERED SFS SHAPE")

    #     gathered_oris = [torch.zeros_like(oris) for _ in range(num_gpus)]
    #     dist.all_gather(gathered_oris, oris)
    #     gathered_oris_tensor = torch.cat(gathered_oris, dim = 0)
    #     #print(gathered_oris_tensor.shape, "gathered ORIS SHAPE")

    #     zinb_loss = ZINB_loss(gathered_oris_tensor.squeeze(0),m,d,p,gathered_sfs_tensor.squeeze(0))
    #     # zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
    #     ## Reconstruction_loss
    #     reconstruction_loss = self.rmse * rmse_loss + self.zinb * zinb_loss
        
    #     return contrastive_loss + reconstruction_loss, contrastive_loss, rmse_loss, zinb_loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
