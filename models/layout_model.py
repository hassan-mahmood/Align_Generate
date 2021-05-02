import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d
import numpy as np
from tqdm import tqdm 
import sg2im.box_utils as box_utils
from sg2im.graph import GraphTripleConv, GraphTripleConvNet
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.layers import build_mlp

BatchNorm = SynchronizedBatchNorm2d

 # def generator(self, z):
 #    with tf.variable_scope("generator") as scope:
 #      gnet = tf.reshape(z, [self.batch_size, self.num_bounding_boxes, 1, 4+self.num_classes])
        #                           B                   H               W        C
 #      h0_0 = self.g_bn0_0(conv2d(gnet, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_0'))
 #      h0_1 = tf.nn.relu(self.g_bn0_1(conv2d(gnet, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_1')))
 #      h0_2 = tf.nn.relu(self.g_bn0_2(conv2d(h0_1, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_2')))
 #      h0_3 = self.g_bn0_3(conv2d(h0_2, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0_3'))
 #      gnet = tf.nn.relu(tf.add(h0_0, h0_3))

 #      gnet = tf.reshape(gnet, [self.batch_size, self.num_bounding_boxes, 1, 1024])
 #      gnet = tf.nn.relu(self.g_bn_x1( tf.add(gnet, self.g_bn_x0(relation_nonLocal(gnet, name='g_non0')))))
 #      gnet = tf.nn.relu(self.g_bn_x3( tf.add(gnet, self.g_bn_x2(relation_nonLocal(gnet, name='g_non2')))))

 #      h1_0 = self.g_bn1_0(conv2d(gnet, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_0'))
 #      h1_1 = tf.nn.relu(self.g_bn1_1(conv2d(h1_0, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_1')))
 #      h1_2 = tf.nn.relu(self.g_bn1_2(conv2d(h1_1, 256, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_2')))
 #      h1_3 = self.g_bn1_3(conv2d(h1_2, 1024, k_h=1, k_w=1, d_h=1, d_w=1, name='g_h1_3'))
 #      gnet = tf.nn.relu(tf.add(h1_0, h1_3))

     
 #      cls_score = conv2d(gnet, 4+self.num_classes, k_h=1, k_w=1, d_h=1, d_w=1, name='cls_score')
 #      cls_score = tf.sigmoid(tf.reshape(cls_score, [-1, self.num_bounding_boxes, 4+self.num_classes]))
 #      # Hassan: Try adding class probabilities here as well (as compared to bounding box coordinates only)
 #      #final_pred = tf.concat([cls_score, cls_prob], axis=-1)
 #      final_pred=cls_score

 #      return final_pred 


# class Layout2Image(nn.Module):
#     def __init__(self,batch_size,num_bounding_boxes,num_classes):
#         super(Layout2Image,self).__init__()
#         self.batch_size=batch_size
#         self.num_bounding_boxes=num_bounding_boxes
#         self.num_classes=num_classes
#         self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)


#     def _build_mask_net(self, num_objs, dim, mask_size):
#         output_dim = 1
#         layers, cur_size = [], 1
#         while cur_size < mask_size:
#           layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
#           layers.append(nn.BatchNorm2d(dim))
#           layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
#           layers.append(nn.ReLU())
#           cur_size *= 2
#         if cur_size != mask_size:
#           raise ValueError('Mask size must be a power of 2')
#         layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
#         return nn.Sequential(*layers)

#     def forward(self,obj_vecs):

#         boxes_pred = self.box_net(obj_vecs)

#         # masks_pred = None
#         # if self.mask_net is not None:
#         #   mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
#         #   masks_pred = mask_scores.squeeze(1).sigmoid()

#         # s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
#         # s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
#         # rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
#         # rel_scores = self.rel_aux_net(rel_aux_input)

#         # H, W = self.image_size
#         # layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

#         # if masks_pred is None:
#         #   layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
#         # else:
#         #   layout_masks = masks_pred if masks_gt is None else masks_gt
#         #   layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
#         #                            obj_to_img, H, W)

#         # if self.layout_noise_dim > 0:
#         #   N, C, H, W = layout.size()
#         #   noise_shape = (N, self.layout_noise_dim, H, W)
#         #   layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
#         #                              device=layout.device)
#         #   layout = torch.cat([layout, layout_noise], dim=1)
#         # img = self.refinement_net(layout)
#         # return img, boxes_pred, masks_pred, rel_scores



class LayoutGenerator(nn.Module):
    def __init__(self,batch_size,num_bounding_boxes,num_classes):
        super(LayoutGenerator,self).__init__()
        self.batch_size=batch_size
        self.num_bounding_boxes=num_bounding_boxes
        self.num_classes=num_classes

        
        #self.g_bn0_0 = nn.Conv2d(188,1024,(188,1,1),stride=(0,0,1),padding=[0,0,0],dilation=[0,0,0])
        self.conv0_0 = nn.Conv2d(4+self.num_classes,1024,(1,1))
        self.g_bn0_0 = BatchNorm(1024)

        self.conv0_1 = nn.Conv2d(4+self.num_classes,256,(1,1))
        self.g_bn0_1= BatchNorm(256)

        self.conv0_2 = nn.Conv2d(256,256,(1,1))
        self.g_bn0_2= BatchNorm(256)

        self.conv0_3 = nn.Conv2d(256,1024,(1,1))
        self.g_bn0_3= BatchNorm(1024)

        self.relu=nn.ReLU()

        self.attention0=Attention()
        self.g_bn_x0=BatchNorm(1024)
        self.g_bn_z0=BatchNorm(1024)

        self.attention1=Attention()
        self.g_bn_x1=BatchNorm(1024)
        self.g_bn_z1=BatchNorm(1024)

        self.attention2=Attention()
        self.g_bn_x2=BatchNorm(1024)
        self.g_bn_z2=BatchNorm(1024)

        self.attention3=Attention()
        self.g_bn_x3=BatchNorm(1024)
        self.g_bn_z3=BatchNorm(1024)

        self.conv1_0 = nn.Conv2d(1024,1024,(1,1))
        self.g_bn1_0 = BatchNorm(1024)

        self.conv1_1 = nn.Conv2d(1024,256,(1,1))
        self.g_bn1_1= BatchNorm(256)

        self.conv1_2 = nn.Conv2d(256,256,(1,1))
        self.g_bn1_2= BatchNorm(256)

        self.conv_fv=nn.Conv2d(256,128,(1,1))

        self.conv1_3 = nn.Conv2d(256,1024,(1,1))
        self.g_bn1_3= BatchNorm(1024)



        self.final_conv=nn.Conv2d(1024,4+self.num_classes,(1,1))
        self.sigmoid=nn.Sigmoid()

    def forward(self,z):
        #gnet=torch.reshape(z,[self.batch_size,self.num_bounding_boxes,1,4+self.num_classes])
        gnet=torch.reshape(z,[self.batch_size,188,self.num_bounding_boxes,1])
        h0_0=self.g_bn0_0(self.conv0_0(gnet))
        h0_1=self.relu(self.g_bn0_1(self.conv0_1(gnet)))
        h0_2=self.relu(self.g_bn0_2(self.conv0_2(h0_1)))
        h0_3=self.g_bn0_3(self.conv0_3(h0_2))
        gnet=self.relu(torch.add(h0_0,h0_3))

        # [5, 1024, 20, 1]
        gnet = self.relu(self.g_bn_z0(torch.add(gnet,self.g_bn_x0(self.attention0(gnet)))))
        gnet = self.relu(self.g_bn_z1(torch.add(gnet,self.g_bn_x1(self.attention1(gnet)))))
        gnet = self.relu(self.g_bn_z2(torch.add(gnet,self.g_bn_x2(self.attention2(gnet)))))
        gnet = self.relu(self.g_bn_z3(torch.add(gnet,self.g_bn_x3(self.attention3(gnet)))))

        h1_0=self.g_bn1_0(self.conv1_0(gnet))
        h1_1=self.relu(self.g_bn1_1(self.conv1_1(h1_0)))

        h1_2=self.relu(self.g_bn1_2(self.conv1_2(h1_1)))

        obj_feature_vector=self.conv_fv(h1_2)

        #h1_2=self.relu(self.g_bn1_2(obj_feature_vector))
        h1_3=self.g_bn1_3(self.conv1_3(h1_2))
        gnet=self.relu(torch.add(h1_0,h1_3))

        output=torch.reshape(self.final_conv(gnet),[-1,self.num_bounding_boxes,4+self.num_classes])
        #output=self.sigmoid(torch.reshape(self.final_conv(gnet),[-1,4+self.num_classes,self.num_bounding_boxes])) 
        #output=self.sigmoid(self.final_conv(gnet))
        obj_feature_vector=torch.reshape(obj_feature_vector,[-1,self.num_bounding_boxes,128])

        return obj_feature_vector,output

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        output_dims=1024
        self.f_v=nn.Conv2d(output_dims,output_dims,(1,1)) 
        self.f_k=nn.Conv2d(output_dims,output_dims,(1,1)) 
        self.f_q=nn.Conv2d(output_dims,output_dims,(1,1)) 
        self.f_r = nn.Conv2d(output_dims,output_dims,(1,1))

    def forward(self,x):
        
        N,C,H,W = x.shape
        # Tensorflow: bhwc, torch: bchw
        f_v=self.f_v(x)
        f_k=self.f_k(x)
        f_q=self.f_q(x)

        f_k=torch.reshape(f_k,[N,C,H*W])
        f_q=torch.reshape(f_q,[N,C,H*W]).permute(0,2,1)
        w=torch.matmul(f_k,f_q)

        f_r = torch.matmul(w.permute(0,2,1),torch.reshape(f_v,[N,C,H*W]))
        f_r = torch.reshape(f_r, [N,C,H,W])
        f_r=self.f_r(f_r)
        return f_r

class LayoutDiscriminator(nn.Module):
    def __init__(self,batch_size,num_bounding_boxes,num_classes,output_height,output_width):
        super(LayoutDiscriminator,self).__init__()
        self.batch_size=batch_size
        self.num_bounding_boxes=num_bounding_boxes
        self.num_classes=num_classes
        self.relu=nn.ReLU()
        self.cls_num=0
        self.height=output_height
        self.width=output_width
        #output_channels=4+num_classes
        output_channels=num_bounding_boxes

        self.conv1=nn.Conv2d(num_bounding_boxes,output_channels,kernel_size=3,stride=2)
        self.bn1=nn.BatchNorm2d(output_channels)
        self.conv2=nn.Conv2d(output_channels,output_channels*2,kernel_size=3,stride=2)
        self.bn2=nn.BatchNorm2d(output_channels*2)
        self.conv3=nn.Conv2d(output_channels*2,output_channels*2,kernel_size=3,stride=2)
        self.bn3=nn.BatchNorm2d(output_channels*2)
        self.fc1=nn.Linear(output_channels*2*49,output_channels) #2*29*7*7
        self.fc2=nn.Linear(output_channels,1)


    def forward(self,x):
        #print('input:',x.shape)
        layout=self.layout_bbox(x,self.batch_size,self.num_bounding_boxes,self.num_classes,self.height,self.width)  
        #print('output layout',layout.shape)
        out=F.relu(self.bn1(self.conv1(layout)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn3(self.conv3(out)))
        out=out.reshape(out.shape[0],-1)
        out=self.relu(self.fc1(out))
        out=self.fc2(out)
        out=F.sigmoid(out)

        return out


  # def discriminator(self, image, reuse=False):

  #   with tf.variable_scope("discriminator") as scope:
  #     if reuse:
  #       scope.reuse_variables()

  #     #layout = layout_point(image, 28, 28, name='layout')
      
  #     layout=layout_bbox(image,self.batch_size,self.num_bounding_boxes,self.num_classes,64,64)
  #     # For bbox layout generation
  #     # layout = layout_bbox(image, 60, 40, name='layout')
      
  #     net = lrelu(self.d_bn0(conv2d(layout, 32, k_h=5, k_w=5, d_h=2, d_w=2, padding='VALID', name='conv1')))
  #     net = lrelu(self.d_bn1(conv2d(net, 64, k_h=5, k_w=5, d_h=2, d_w=2, padding='VALID', name='conv2')))
  #     net = tf.reshape(net, [self.batch_size, -1])      
  #     net = lrelu(self.d_bn2(linear(net, 512, scope='fc2')))
  #     net = linear(net, 1, 'fc3')
    
  #   return tf.nn.sigmoid(net), net
    def layout_bbox(self,final_pred, batch_size,num_bboxes,num_classes, output_height, output_width):
        # 5, 188, 20
        final_pred = torch.reshape(final_pred,[batch_size,num_bboxes,4+num_classes])
        #print('Final pred:',final_pred.shape)
        return self.rectangle_render(final_pred)
        0/0
        final_pred = torch.reshape(final_pred,[batch_size,4+num_classes,num_bboxes])
        print('Final pred requires grad:',final_pred.requires_grad)
        bbox_reg = final_pred[:,:4,:]
        cls_prob = final_pred[:,4:,:]

        print('bbox requires grad:',bbox_reg.requires_grad)
        bbox_reg=torch.reshape(bbox_reg,[batch_size,num_bboxes,4])

        x_c=bbox_reg[:,:,0] * output_width
        y_c = bbox_reg[:,:,1] * output_height
        w = bbox_reg[:,:,2] * output_width
        h = bbox_reg[:,:,3] * output_height

        x1 = x_c - 0.5*w
        x2 = x_c + 0.5*w
        y1 = y_c - 0.5*h
        y2 = y_c + 0.5*h 


        xt=torch.reshape(torch.range(start=0,end=output_width,dtype=torch.float32),[1,1,1,-1])
        xt = torch.reshape(torch.tile(xt,[batch_size,num_bboxes,output_height,1]),[batch_size,num_bboxes,-1])

        yt=torch.reshape(torch.range(start=0,end=output_height,dtype=torch.float32),[1,1,1,-1])
        yt = torch.reshape(torch.tile(yt,[batch_size,num_bboxes,1,output_width]),[batch_size,num_bboxes,-1])

        x1_diff=torch.reshape(xt-x1, [batch_size,num_bboxes,output_height,output_width,1])
        y1_diff=torch.reshape(yt-y1, [batch_size,num_bboxes,output_height,output_width,1])
        x2_diff=torch.reshape(x2-xt, [batch_size,num_bboxes,output_height,output_width,1])
        y2_diff=torch.reshape(y2-yt, [batch_size,num_bboxes,output_height,output_width,1])

        x1_line=self.relu(1.0 - torch.abs(x1_diff)) * torch.minimum(self.relu(y1_diff),1.0) * torch.minimum(self.relu(y2_diff),1.0)
        print(x1_line.shape)
        print(x1_line)

        0/0

    def rectangle_render(self,x):
        # I's size [b,c,h,w]
        # x's size [b,num_ele+5,cls_num+4]
        batch_size=x.size(0)
        # wrong
        # I=torch.zeros((batch_size,self.num_elements,self.height,self.width))
        h_index=torch.arange(0,self.height).cuda()
        w_index=torch.arange(0,self.width).cuda()
        hh=h_index.repeat(len(w_index))
        ww=w_index.view(-1,1).repeat(1,len(h_index)).view(-1)
        index=torch.stack([ww,hh],dim=-1) #[[0,0],[0,1]...[ww-1,hh-1]]
        index_=index.unsqueeze(0).repeat(batch_size,1,1)
        index_col=index_[:,:,0]
        index_row=index_[:,:,1]
        x_trans=x.permute(0,2,1)
        index_col=index_col.unsqueeze(2)
        index_row=index_row.unsqueeze(2)
        sub_xL=index_col-x_trans[:,self.cls_num,:].unsqueeze(1).long()
        sub_yT=index_row-x_trans[:,self.cls_num+1,:].unsqueeze(1).long()
        sub_xR=index_col-x_trans[:,self.cls_num+2,:].unsqueeze(1).long()
        sub_yB=index_row-x_trans[:,self.cls_num+3,:].unsqueeze(1).long()
        sub_y=x_trans[:,self.cls_num+3,:].unsqueeze(1).long()-index_row
        sub_x=x_trans[:,self.cls_num+2,:].unsqueeze(1).long()-index_col
        tmp1=F.relu(sub_yT)
        tmp1[tmp1>1]=1
        tmp2=F.relu(sub_y)
        tmp2[tmp2>1]=1
        F_0=F.relu(1-torch.abs(sub_xL))*tmp1*tmp2
        F_1=F.relu(1-torch.abs(sub_xR))*tmp1*tmp2
        tmp1 = F.relu(sub_xL)
        tmp1[tmp1 > 1] = 1
        tmp2 = F.relu(sub_x)
        tmp2[tmp2 > 1] = 1
        F_2=F.relu(1-torch.abs(sub_yT))*tmp1*tmp2
        F_3=F.relu(1-torch.abs(sub_yB))*tmp1*tmp2


        # val shape [batch_size,hei*wid,num_elem]
        val,index_ftheta=torch.max(torch.stack((F_0,F_1,F_2,F_3),dim=2),dim=2)
        #print('val shape:',val.shape)

        x_prob=x[:,:,4:]
        #print('x prob:',x_prob.shape)
        x_prob=x_prob.unsqueeze(1)#[batch_size,1,num_elem,cls_num]

        F_theta=val.unsqueeze(3).float() #[batch_szie,hei*wid,num_elem,1]
        #print('F_theta:',F_theta.shape)

        prod=x_prob*F_theta #[batch_szie,hei*wid,num_elem,cls_num]
        #print('prod:',prod.shape)
        res,index_res=torch.max(prod,3)
        #print('res:',res.shape)
        I=res.contiguous().view(batch_size,self.height,self.width,-1).permute(0,1,3,2)
        I=I.permute(0,2,1,3)
        return I


        


    # def layout_bbox(final_pred, batch_size,num_bboxes,num_classes, output_height, output_width, name="layout_bbox"):
    #     with tf.variable_scope(name):

    #         #num_classes are 184. So first four are bounding box coordinates and the rest 184 are class probability
    #         final_pred = tf.reshape(final_pred, [batch_size, num_bboxes, 188])

    #         bbox_reg = tf.slice(final_pred, [0, 0, 0], [-1, -1, 4])
    #         cls_prob = tf.slice(final_pred, [0, 0, 4], [-1, -1, num_classes])

    #         bbox_reg = tf.reshape(bbox_reg, [batch_size, num_bboxes, 4])

    #         x_c = tf.slice(bbox_reg, [0, 0, 0], [-1, -1, 1]) * output_width
    #         y_c = tf.slice(bbox_reg, [0, 0, 1], [-1, -1, 1]) * output_height
    #         w   = tf.slice(bbox_reg, [0, 0, 2], [-1, -1, 1]) * output_width
    #         h   = tf.slice(bbox_reg, [0, 0, 3], [-1, -1, 1]) * output_height

    #         x1 = x_c - 0.5*w
    #         x2 = x_c + 0.5*w
    #         y1 = y_c - 0.5*h
    #         y2 = y_c + 0.5*h

    #         xt = tf.reshape(tf.range(output_width, dtype=tf.float32), [1, 1, 1, -1])
    #         xt = tf.reshape(tf.tile(xt, [batch_size, num_bboxes, output_height, 1]), [batch_size, num_bboxes, -1])

    #         yt = tf.reshape(tf.range(output_height, dtype=tf.float32), [1, 1, -1, 1])
    #         yt = tf.reshape(tf.tile(yt, [batch_size, num_bboxes, 1, output_width]), [batch_size, num_bboxes, -1])

    #         x1_diff = tf.reshape(xt-x1, [batch_size, num_bboxes, output_height, output_width, 1])
    #         y1_diff = tf.reshape(yt-y1, [batch_size, num_bboxes, output_height, output_width, 1])
    #         x2_diff = tf.reshape(x2-xt, [batch_size, num_bboxes, output_height, output_width, 1])
    #         y2_diff = tf.reshape(y2-yt, [batch_size, num_bboxes, output_height, output_width, 1])

    #         x1_line = tf.nn.relu(1.0 - tf.abs(x1_diff)) * tf.minimum(tf.nn.relu(y1_diff), 1.0) * tf.minimum(tf.nn.relu(y2_diff), 1.0)
    #         x2_line = tf.nn.relu(1.0 - tf.abs(x2_diff)) * tf.minimum(tf.nn.relu(y1_diff), 1.0) * tf.minimum(tf.nn.relu(y2_diff), 1.0)
    #         y1_line = tf.nn.relu(1.0 - tf.abs(y1_diff)) * tf.minimum(tf.nn.relu(x1_diff), 1.0) * tf.minimum(tf.nn.relu(x2_diff), 1.0)
    #         y2_line = tf.nn.relu(1.0 - tf.abs(y2_diff)) * tf.minimum(tf.nn.relu(x1_diff), 1.0) * tf.minimum(tf.nn.relu(x2_diff), 1.0)

    #         xy_max = tf.reduce_max(tf.concat([x1_line, x2_line, y1_line, y2_line], axis=-1), axis=-1, keep_dims=True)

    #         spatial_prob = tf.multiply(tf.tile(xy_max, [1, 1, 1, 1, num_classes]), tf.reshape(cls_prob, [batch_size, num_bboxes, 1, 1, num_classes]))
    #         spatial_prob_max = tf.reduce_max(spatial_prob, axis=1, keep_dims=False)
    #         print('\n\nLayout wireframe called')

    #         return spatial_prob_max


# class ResnetGenerator128(nn.Module):
#     def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
#         super(ResnetGenerator128, self).__init__()
#         self.num_classes = num_classes

#         self.label_embedding = nn.Embedding(num_classes, 180)

#         num_w = 128 + 180
#         self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

#         self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
#         self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
#         self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
#         self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w, psp_module=True)
#         self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
#         self.final = nn.Sequential(BatchNorm(ch),
#                                    nn.ReLU(),
#                                    conv2d(ch, output_dim, 3, 1, 1),
#                                    nn.Tanh())
#         print('Creating resnet128')
#         # mapping function
#         mapping = list()
#         self.mapping = nn.Sequential(*mapping)

#         self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

#         self.sigmoid = nn.Sigmoid()
#         print('Creating resnet1282')
#         self.mask_regress = MaskRegressNetv2(num_w)
#         print('Creating resnet128')

#         self.init_parameter()
#         print('Creating resnet128 done')

#     def forward(self, z, bbox, z_im=None, y=None):
#         b, o = z.size(0), z.size(1)
#         label_embedding = self.label_embedding(y)

#         z = z.view(b * o, -1)
#         label_embedding = label_embedding.view(b * o, -1)

#         latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

#         w = self.mapping(latent_vector.view(b * o, -1))
#         # preprocess bbox
#         bmask = self.mask_regress(w, bbox)

#         if z_im is None:
#             z_im = torch.randn((b, 128), device=z.device)

#         bbox_mask_ = bbox_mask(z, bbox, 64, 64)

#         # 4x4
#         x = self.fc(z_im).view(b, -1, 4, 4)
#         # 8x8
#         x, stage_mask = self.res1(x, w, bmask)

#         # 16x16
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
#         x, stage_mask = self.res2(x, w, stage_bbox)

#         # 32x32
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
#         x, stage_mask = self.res3(x, w, stage_bbox)

#         # 64x64
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
#         x, stage_mask = self.res4(x, w, stage_bbox)

#         # 128x128
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
#         x, _ = self.res5(x, w, stage_bbox)

#         # save_path1 = 'samples/tmp/edit/apponly/1292_bmask2_0.npy'
#         # save_path2 = 'samples/tmp/edit/apponly/1292_stage2_bbox_0.npy'
#         # np.save(save_path1, bmask.cpu().detach().numpy())
#         # np.save(save_path2, stage_bbox.cpu().detach().numpy())

#         # to RGB
#         x = self.final(x)
#         return x

#     def init_parameter(self):
#         for k in tqdm(self.named_parameters()):
#             if k[1].dim() > 1:
#                 torch.nn.init.orthogonal_(k[1])
#             if k[0][-4:] == 'bias':
#                 torch.nn.init.constant_(k[1], 0)


# class ResnetGenerator256(nn.Module):
#     def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
#         super(ResnetGenerator256, self).__init__()
#         self.num_classes = num_classes

#         self.label_embedding = nn.Embedding(num_classes, 180)

#         num_w = 128 + 180
#         self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

#         self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
#         self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
#         self.res3 = ResBlock(ch * 8, ch * 8, upsample=True, num_w=num_w)
#         self.res4 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
#         self.res5 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w)
#         self.res6 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
#         self.final = nn.Sequential(BatchNorm(ch),
#                                    nn.ReLU(),
#                                    conv2d(ch, output_dim, 3, 1, 1),
#                                    nn.Tanh())

#         # mapping function
#         mapping = list()
#         self.mapping = nn.Sequential(*mapping)

#         self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha5 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.sigmoid = nn.Sigmoid()

#         self.mask_regress = MaskRegressNetv2(num_w)
#         self.init_parameter()

#     def forward(self, z, bbox, z_im=None, y=None, include_mask_loss=False):
#         b, o = z.size(0), z.size(1)

#         label_embedding = self.label_embedding(y)

#         z = z.view(b * o, -1)
#         label_embedding = label_embedding.view(b * o, -1)

#         latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

#         w = self.mapping(latent_vector.view(b * o, -1))

#         # preprocess bbox
#         bmask = self.mask_regress(w, bbox)

#         if z_im is None:
#             z_im = torch.randn((b, 128), device=z.device)

#         bbox_mask_ = bbox_mask(z, bbox, 128, 128)

#         latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
#         w = self.mapping(latent_vector.view(b * o, -1))

#         # 4x4
#         x = self.fc(z_im).view(b, -1, 4, 4)
#         # 8x8
#         # label mask
#         x, stage_mask = self.res1(x, w, bmask)

#         # 16x16
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
#         x, stage_mask = self.res2(x, w, stage_bbox)

#         # 32x32
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

#         alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
#         x, stage_mask = self.res3(x, w, stage_bbox)

#         # 64x64
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

#         alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
#         x, stage_mask = self.res4(x, w, stage_bbox)

#         # 128x128
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

#         alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
#         x, stage_mask = self.res5(x, w, stage_bbox)

#         # 256x256
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

#         alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
#         x, _ = self.res6(x, w, stage_bbox)
#         # to RGB
#         x = self.final(x)
#         return x

#     def init_parameter(self):
#         for k in self.named_parameters():
#             if k[1].dim() > 1:
#                 torch.nn.init.orthogonal_(k[1])
#             if k[0][-4:] == 'bias':
#                 torch.nn.init.constant_(k[1], 0)


# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128, predict_mask=True, psp_module=False):
#         super(ResBlock, self).__init__()
#         self.upsample = upsample
#         self.h_ch = h_ch if h_ch else out_ch
#         self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
#         self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
#         self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w, batchnorm_func=BatchNorm)
#         self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w, batchnorm_func=BatchNorm)
#         self.learnable_sc = in_ch != out_ch or upsample
#         if self.learnable_sc:
#             self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
#         self.activation = nn.ReLU()

#         self.predict_mask = predict_mask
#         if self.predict_mask:
#             if psp_module:
#                 self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
#                                                nn.Conv2d(100, 184, kernel_size=1))
#             else:
#                 self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
#                                                BatchNorm(100),
#                                                nn.ReLU(),
#                                                nn.Conv2d(100, 184, 1, 1, 0, bias=True))

#     def residual(self, in_feat, w, bbox):
#         x = in_feat
#         x = self.b1(x, w, bbox)
#         x = self.activation(x)
#         if self.upsample:
#             x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.conv1(x)
#         x = self.b2(x, w, bbox)
#         x = self.activation(x)
#         x = self.conv2(x)
#         return x

#     def shortcut(self, x):
#         if self.learnable_sc:
#             if self.upsample:
#                 x = F.interpolate(x, scale_factor=2, mode='nearest')
#             x = self.c_sc(x)
#         return x

#     def forward(self, in_feat, w, bbox):
#         out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
#         if self.predict_mask:
#             mask = self.conv_mask(out_feat)
#         else:
#             mask = None
#         return out_feat, mask


# def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
#     conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
#     if spectral_norm:
#         return nn.utils.spectral_norm(conv, eps=1e-4)
#     else:
#         return conv


# def batched_index_select(input, dim, index):
#     expanse = list(input.shape)
#     expanse[0] = -1
#     expanse[dim] = -1
#     index = index.expand(expanse)
#     return torch.gather(input, dim, index)


# def bbox_mask(x, bbox, H, W):
#     b, o, _ = bbox.size()
#     N = b * o

#     bbox_1 = bbox.float().view(-1, 4)
#     x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
#     ww, hh = bbox_1[:, 2], bbox_1[:, 3]

#     x0 = x0.contiguous().view(N, 1).expand(N, H)
#     ww = ww.contiguous().view(N, 1).expand(N, H)
#     y0 = y0.contiguous().view(N, 1).expand(N, W)
#     hh = hh.contiguous().view(N, 1).expand(N, W)

#     X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
#     Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

#     X = (X - x0.to(X.device)) / ww.to(X.device)
#     Y = (Y - y0.to(Y.device)) / hh.to(Y.device)

#     X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
#     Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

#     out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
#     return out_mask.view(b, o, H, W)


# class PSPModule(nn.Module):
#     """
#     Reference:
#         Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
#     """

#     def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
#         super(PSPModule, self).__init__()

#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
#             BatchNorm(out_features),
#             nn.ReLU(),
#             nn.Dropout2d(0.1)
#         )

#     def _make_stage(self, features, out_features, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
#         bn = nn.BatchNorm2d(out_features)
#         return nn.Sequential(prior, conv, bn, nn.ReLU())

#     def forward(self, feats):
#         h, w = feats.size(2), feats.size(3)
#         priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
#         bottle = self.bottleneck(torch.cat(priors, 1))
#         return bottle
