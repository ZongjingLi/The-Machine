import torch
import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def swish(x):
    return x * torch.sigmoid(x)

class CondResBlock(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, latent_grid=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.latent_grid = latent_grid

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=False)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm2d(filters, affine=False)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=False)


        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, latent):
        x_orig = x

        latent_1 = self.latent_fc1(latent)
        latent_2 = self.latent_fc2(latent)

        gain = latent_1[:, :self.filters, None, None]
        bias = latent_1[:, self.filters:, None, None]

        gain2 = latent_2[:, :self.filters, None, None]
        bias2 = latent_2[:, self.filters:, None, None]

        x = self.conv1(x)
        x = gain * x + bias
        x = swish(x)


        x = self.conv2(x)
        x = gain2 * x + bias2
        x = swish(x)

        x_out = x_orig + x

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out


class CondResBlockNoLatent(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, upsample=False):
        super(CondResBlockNoLatent, self).__init__()

        self.filters = filters
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.GroupNorm(int(32  * filters / 128), filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.GroupNorm(int(32 * filters / 128), filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        self.upsample = upsample
        self.upsample_module = nn.Upsample(scale_factor=2)
        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        if upsample:
            self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_orig = x


        x = self.conv1(x)
        x = swish(x)

        x = self.conv2(x)
        x = swish(x)

        x_out = x_orig + x

        if self.upsample:
            x_out = self.upsample_module(x_out)
            x_out = swish(self.conv_downsample(x_out))

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out

class BroadcastConvDecoder(nn.Module):
    def __init__(self, im_size, latent_dim):
        super().__init__()
        self.im_size = im_size + 8
        self.latent_dim = latent_dim
        self.init_grid()

        self.g = nn.Sequential(
                    nn.Conv2d(self.latent_dim+2, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, self.latent_dim, 1, 1, 0)
                    )

    def init_grid(self):
        x = torch.linspace(0, 1, self.im_size)
        y = torch.linspace(0, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)


    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = self.broadcast(z)
        x = self.g(z)
        return x


class LatentEBM(nn.Module):
    def __init__(self, args):
        super(LatentEBM, self).__init__()

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim

        self.components = args.components

        self.pos_embed = args.pos_embed

        if self.pos_embed:
            self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv1_embed = nn.Conv2d(2, filter_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        self.gain = nn.Linear(args.latent_dim, filter_dim)
        self.bias = nn.Linear(args.latent_dim, filter_dim)

        self.recurrent_model = args.recurrent_model


        self.im_size = 32

        self.layer_encode = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer1 = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer2 = CondResBlock(filters=filter_dim, latent_dim=latent_dim)
        self.mask_decode = BroadcastConvDecoder(64, latent_dim)

        self.latent_map = nn.Linear(latent_dim, filter_dim * 8)
        self.energy_map = nn.Linear(filter_dim * 2, 1)

        self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)

        self.decode_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)
        self.decode_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)
        self.decode_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)

        self.latent_decode = nn.Conv2d(filter_dim, latent_dim_expand, kernel_size=3, stride=1, padding=1)

        self.downsample = Downsample(channels=args.latent_dim)

        if self.recurrent_model:
            self.embed_layer4 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
            self.lstm = nn.LSTM(filter_dim, filter_dim, 1)
            self.embed_fc2 = nn.Linear(filter_dim, latent_dim)

            self.at_fc1 = nn.Linear(filter_dim*2, filter_dim)
            self.at_fc2 = nn.Linear(filter_dim, 1)

            self.map_embed = nn.Linear(filter_dim*2, filter_dim)

            self.pos_embedding = nn.Parameter(torch.zeros(16, filter_dim))
        else:
            self.embed_fc1 = nn.Linear(filter_dim, filter_dim)
            self.embed_fc2 = nn.Linear(filter_dim, latent_dim_expand)

        self.init_grid()

    def gen_mask(self, latent):
        return self.mask_decode(latent)

    def init_grid(self):
        x = torch.linspace(0, 1, self.im_size)
        y = torch.linspace(0, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)

    def embed_latent(self, im):
        x = self.embed_conv1(im)
        x = F.relu(x)
        x = self.embed_layer1(x)
        x = self.embed_layer2(x)
        x = self.embed_layer3(x)

        if self.recurrent_model:

            x = self.embed_layer4(x)

            s = x.size()
            x = x.view(s[0], s[1], -1)
            x = x.permute(0, 2, 1).contiguous()

            h = torch.zeros(1, im.size(0), self.filter_dim).to(x.device), torch.zeros(1, im.size(0), self.filter_dim).to(x.device)
            outputs = []

            for i in range(self.components):
                (sx, cx) = h

                cx = cx.permute(1, 0, 2).contiguous()
                context = torch.cat([cx.expand(-1, x.size(1), -1), x], dim=-1)
                at_wt = self.at_fc2(F.relu(self.at_fc1(context)))
                at_wt = F.softmax(at_wt, dim=1)
                inp = (at_wt * context).sum(dim=1, keepdim=True)
                inp = self.map_embed(inp)
                inp = inp.permute(1, 0, 2).contiguous()

                output, h = self.lstm(inp, h)
                outputs.append(output)

            output = torch.cat(outputs, dim=0)
            output = output.permute(1, 0, 2).contiguous()
            output = self.embed_fc2(output)
            s = output.size()
            output = output.view(s[0], -1)
        else:
            x = x.mean(dim=2).mean(dim=2)

            x = x.view(x.size(0), -1)
            output = self.embed_fc1(x)
            x = F.relu(self.embed_fc1(x))
            output = self.embed_fc2(x)

        return output

    def forward(self, x, latent):

        if self.pos_embed:
            b = x.size(0)
            x_grid = self.x_grid.expand(b, 1, -1, -1).to(x.device)
            y_grid = self.y_grid.expand(b, 1, -1, -1).to(x.device)
            coord_grid = torch.cat([x_grid, y_grid], dim=1)

        # x = x.contiguous()
        inter = self.conv1(x)
        inter = swish(inter)

        if self.pos_embed:
            pos_inter = self.conv1_embed(coord_grid)
            pos_inter = swish(pos_inter)

            inter = torch.cat([inter, pos_inter], dim=1)

        x = self.avg_pool(inter)

        x = self.layer_encode(x, latent)

        x = self.layer1(x, latent)
        x = self.layer2(x, latent)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)

        energy = self.energy_map(x)

        return energy

import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--device",default = device)
parser.add_argument('--filter_dim', default=16, type=int, help='number of filters to use')
parser.add_argument('--components', default=4, type=int, help='number of components to explain an image with')
parser.add_argument('--component_weight', action='store_true', help='optimize for weights of the components also')
parser.add_argument('--tie_weight', default = True,action='store_true', help='tie the weights between seperate models')
parser.add_argument('--optimize_mask', action='store_true', help='also optimize a segmentation mask over image')
parser.add_argument('--recurrent_model', default = True,action='store_true', help='use a recurrent model to infer latents')
parser.add_argument('--pos_embed',default = True, action='store_true', help='add a positional embedding to model')
parser.add_argument('--spatial_feat', action='store_true', help='use spatial latents for object segmentation')


parser.add_argument('--num_steps', default=10, type=int, help='Steps of gradient descent for training')
parser.add_argument('--num_visuals', default=16, type=int, help='Number of visuals')
parser.add_argument('--num_additional', default=0, type=int, help='Number of additional components to add')

parser.add_argument('--step_lr', default=500.0, type=float, help='step size of latents')

parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--sample', action='store_true', help='generate negative samples through Langevin')
parser.add_argument('--decoder', action='store_true', help='decoder for model')

ebm_config = parser.parse_args(args = [])


def gen_image(latents,models,im,im_neg,num_steps):
    im_noise = torch.randn_like(im_neg).detach()
    im_negs = []

    latents = torch.stack(latents,dim = 0)

    # use the diffusion to generate the lowest energy model
    im_neg.requires_grad_(requires_grad = True)
    s = im.size()
    masks = torch.torch.zeros(s[0], ebm_config.components,s[-2],s[-1]).to(im_neg.device)
    masks.requires_grad_(requires_grad = True)

    for i in range(num_steps):
        im_noise.normal_()

        energy = 0
        for j in range(len(latents)):
            energy = energy + models[j %  ebm_config.components].forward(im_neg,latents[j])
        
        im_grad, = torch.autograd.grad([energy.sum()],[im_neg],create_graph = True)

        im_neg = im_neg - 1000.0 * im_grad + im_noise * 0.1

        latents = latents

        im_neg = torch.clamp(im_neg,0,1)
        im_negs.append(im_neg)
        im_neg = im_neg.detach()
        im_neg.requires_grad_()
    return im_neg,im_negs,im_grad,masks

import matplotlib.pyplot as plt

import tqdm

def train(dataloader,models,optims):
    model  = LatentEBM(ebm_config)
    models = [model for i in range(ebm_config.components)]
    optims = [torch.optim.Adam(model.parameters(),lr = 2e-4) for model in models]

    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    epoch = 5000
    for i in range(epoch):
        total_loss = 0
        itr = 0
        for sample in tqdm.tqdm(dataloader):
            im = sample["image"]

            latents = models[0].embed_latent(im)
            latents = torch.chunk(latents, ebm_config.components, dim=1)
            im_neg = torch.rand_like(im)
            im_neg_init = im_neg

            im_neg, im_negs, im_grad, _ = gen_image(latents,models,im ,im_neg, 10)
            im_negs = torch.stack(im_negs, dim=1)

            energy_pos = 0
            energy_neg = 0

            energy_poss = []
            energy_negs = []
            for i in range(ebm_config.components):
                energy_poss.append(models[i].forward(im, latents[i]))
                energy_negs.append(models[i].forward(im_neg.detach(), latents[i]))

            energy_pos = torch.stack(energy_poss, dim=1)
            energy_neg = torch.stack(energy_negs, dim=1)
            ml_loss = (energy_pos - energy_neg).mean()

            im_loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()

            total_loss = im_loss +  ml_loss
            total_loss.backward()

            [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in models]
            [optimizer.step() for optimizer in optims]
            [optimizer.zero_grad() for optimizer in optims]
            if itr%2 ==0:
              plt.subplot(1,2,1);plt.cla()
              plt.imshow(im_neg.cpu().detach()[0].permute([1,2,0]))
              plt.subplot(1,2,2);plt.cla()
              plt.imshow(im.cpu()[0].permute([1,2,0]))
              plt.pause(0.0001)
            itr+=1

        plt.imshow(im_neg.detach()[0].permute([1,2,0]))
        torch.save(model,"model.ckpt")