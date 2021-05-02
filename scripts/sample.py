#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2 
from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.model import Layout2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager
from models.layout_model import * 
from tqdm import tqdm 

torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()

# Optimization hyperparameters
parser.add_argument('--batch_size', default=1, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)


# VG-specific options
parser.add_argument('--max_objects_per_image', default=8, type=int)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default='/mnt/xfs1/hassan2/projectdata/data/coco/val2017/')
parser.add_argument('--coco_val_image_dir',
         default='/mnt/xfs1/hassan2/projectdata/data/coco/val2017/')
parser.add_argument('--coco_train_instances_json',
         default='/mnt/xfs1/hassan2/projectdata/data/coco/annotations/instances_val2017.json')
parser.add_argument('--coco_train_stuff_json',
         default='/mnt/xfs1/hassan2/projectdata/data/coco/annotations/stuff_val2017.json')
parser.add_argument('--coco_val_instances_json',
         default='/mnt/xfs1/hassan2/projectdata/data/coco/annotations/instances_val2017.json')
parser.add_argument('--coco_val_stuff_json',
         default='/mnt/xfs1/hassan2/projectdata/data/coco/annotations/stuff_val2017.json')

parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)


# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=16, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight 
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight
# Output options

parser.add_argument('--output_folder', default='generated_outputs')
parser.add_argument('--checkpoint_start_from', default='stats/epoch_3_batch_99_with_model.pt')
parser.add_argument('--num_sample_imgs',default=10,type=int)

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  curr_loss = curr_loss * weight
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss


def check_args(args):
  H, W = args.image_size
  for _ in args.refinement_network_dims[1:]:
    H = H // 2
  if H == 0:
    raise ValueError("Too many layers in refinement network")


def build_model(args, vocab):
  kwargs = {
    'vocab': vocab,
    'image_size': args.image_size,
    'embedding_dim': args.embedding_dim,
    'gconv_dim': args.gconv_dim,
    'gconv_hidden_dim': args.gconv_hidden_dim,
    'gconv_num_layers': args.gconv_num_layers,
    'mlp_normalization': args.mlp_normalization,
    'refinement_dims': args.refinement_network_dims,
    'normalization': args.normalization,
    'activation': args.activation,
    'mask_size': args.mask_size,
    'layout_noise_dim': args.layout_noise_dim,
  }
  model = Layout2ImModel(**kwargs)
  return model, kwargs


def build_obj_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_obj_weight = args.d_obj_weight
  if d_weight == 0 or d_obj_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'vocab': vocab,
    'arch': args.d_obj_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
    'object_size': args.crop_size,
  }
  discriminator = AcCropDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


def build_img_discriminator(args, vocab):
  discriminator = None
  d_kwargs = {}
  d_weight = args.discriminator_loss_weight
  d_img_weight = args.d_img_weight
  if d_weight == 0 or d_img_weight == 0:
    return discriminator, d_kwargs

  d_kwargs = {
    'arch': args.d_img_arch,
    'normalization': args.d_normalization,
    'activation': args.d_activation,
    'padding': args.d_padding,
  }
  discriminator = PatchDiscriminator(**d_kwargs)
  return discriminator, d_kwargs


def build_coco_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'stuff_only': args.coco_stuff_only,
    'image_size': args.image_size,
    'mask_size': args.mask_size,
    'max_samples': args.num_train_samples,
    'min_object_size': args.min_object_size,
    'min_objects_per_image': args.min_objects_per_image,
    'instance_whitelist': args.instance_whitelist,
    'stuff_whitelist': args.stuff_whitelist,
    'include_other': args.coco_include_other,
    #'include_relationships': args.include_relationships,
  }
  train_dset = CocoSceneGraphDataset(**dset_kwargs)
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  dset_kwargs['max_samples'] = args.num_val_samples
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

  assert train_dset.vocab == val_dset.vocab
  vocab = json.loads(json.dumps(train_dset.vocab))

  return vocab, train_dset, val_dset


def build_loaders(args):

  vocab, train_dset, val_dset = build_coco_dsets(args)
  collate_fn = coco_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader

def calculate_model_losses(args, model, img, img_pred,
                           bbox, bbox_pred, logit_boxes,generated_boxes,original_combined):
  total_loss = torch.zeros(1).to(img)
  losses = {}

  l1_pixel_weight = args.l1_pixel_loss_weight

  l1_pixel_loss = F.l1_loss(img_pred, img)

  total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                        l1_pixel_weight)

  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args.bbox_pred_loss_weight)

  orig_labels=torch.argmax(original_combined[:,:,4:],dim=2).view(-1)

  #print('logits:',logit_boxes.shape)
  loss_classification=nn.CrossEntropyLoss()
  loss_classify = loss_classification(logit_boxes[:,:,4:].view(-1,184), orig_labels)
  
  total_loss = add_loss(total_loss, loss_classify, losses, 'classification_loss')
  mse_loss = F.mse_loss(generated_boxes[:,:,:4], original_combined[:,:,:4])
  total_loss = add_loss(total_loss, mse_loss, losses, 'mse_loss',
                        args.bbox_pred_loss_weight)

  return total_loss, losses


def main(args):
  #print(args)
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor

  vocab, train_loader, val_loader = build_loaders(args)
  model, model_kwargs = build_model(args, vocab)
  model.type(float_dtype)
  model=model.cuda()

  layoutgen = LayoutGenerator(args.batch_size,args.max_objects_per_image+1,184).cuda()

  if(not os.path.exists(args.output_folder)):
    os.makedirs(args.output_folder)

  if(args.checkpoint_start_from is not None):
    model_path=args.checkpoint_start_from

    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model_state'])
    layoutgen.load_state_dict(checkpoint['layout_gen'])

  num_samples=0
  for batchnum,batch in enumerate(tqdm(val_loader)):
    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, combined,all_num_objs = batch
    imgs,objs,boxes,masks,triples,obj_to_img,triple_to_img,combined,all_num_objs = imgs.cuda(),objs.cuda(),boxes.cuda(),masks.cuda(),triples.cuda(),obj_to_img.cuda(),triple_to_img.cuda(),combined.cuda(),all_num_objs.cuda() 

    for k in range(2):
      zlist = []
      for i in range(args.batch_size):
          geo_z=torch.normal(0,1,size=(args.max_objects_per_image+1,4))
          z=torch.FloatTensor(geo_z)
          zlist.append(z)

      zlist=torch.stack(zlist).cuda()
      zlist=torch.cat((zlist,combined[:,:,4:]),dim=2)
      
      feature_vectors,logit_boxes = layoutgen(zlist.cuda())
      generated_boxes = 1/(1+torch.exp(-1*logit_boxes))
    
      new_gen_boxes = torch.empty((0,4)).cuda()
      new_feature_vecs=torch.empty((0,args.embedding_dim)).cuda()
      #print(generated_boxes[0,:,:4])

      for kb in range(args.batch_size):
          new_gen_boxes=torch.cat([new_gen_boxes,torch.squeeze(generated_boxes[kb,:all_num_objs[kb],:4])],dim=0)
          new_feature_vecs=torch.cat([new_feature_vecs,torch.squeeze(feature_vectors[kb,:all_num_objs[kb],:])],dim=0)

      boxes_pred=new_gen_boxes

      triples=None
      imgs_pred = model(new_feature_vecs,new_gen_boxes, triples, obj_to_img)
      imgs_pred = imagenet_deprocess_batch(imgs_pred)
      
      for idx in range(imgs_pred.shape[0]):
        current_img=imgs_pred[idx,:,:,:].numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join(args.output_folder,str(batchnum)+'_'+str(k)+'_'+str(idx)+'.jpg'), current_img)

      num_samples+=1

    if(num_samples+1>=args.num_sample_imgs):
      break

    
    


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

