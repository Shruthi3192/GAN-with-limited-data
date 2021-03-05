#from google.colab import auth
#auth.authenticate_user()
#from google.colab import drive
from tqdm import tqdm
#!git init
#!git config --global user.email "ashruthi10@gmail.com"
#!git config --global user.name "Shruthi3192"
#!git add "/content/drive/MyDrive/Colab Notebooks/SNGan.ipynb"
#!git commit -m "first commit"
#!git remote add origin https://Shruthi3192:Ashruthi@10@github.com/Shruthi3192/reponame.git
#!git push -u origin Collabcommit1
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
from discriminator import Discriminator
from generator import Generator
from spectral import SpectralNorm


#drive.mount('/content/drive')
import sys
#sys.path.insert(0,'/content/drive/My Drive/Colab Notebooks')
#import Discriminator
#import Generator


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

batch_size=64
loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/', train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5
lr=0.001
generator = Generator(Z_dim).cuda()
discriminator = Discriminator().cuda()
# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #to check for gpu



# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

def train(epoch):
   
    for batch_idx, (data, target) in enumerate(loader):
        print(batch_idx)
        if data.size()[0] != batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            '''
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
            '''
            disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(batch_size, 1).cuda()))
            
            
            disc_loss.backward()
            optim_disc.step()
        

        z = Variable(torch.randn(batch_size, Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        '''
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
        '''
        gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()
       

        #if batch_idx % 100 == 0:
            #print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
    scheduler_d.step()
    scheduler_g.step()

fixed_z = Variable(torch.randn(batch_size, Z_dim).cuda())

def evaluate(epoch):
    
    samples = generator(fixed_z).cpu().data.numpy()[:64]
   

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)
      

    #plt.savefig('/Saveimage/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    #plt.close(fig)

if __name__ == '__main__':

    for epoch in tqdm(range(1000)):
        train(epoch)
        evaluate(epoch)
    #torch.save(discriminator.state_dict(), '/'.format(epoch))
    #torch.save(generator.state_dict(), '/'.format(epoch))
