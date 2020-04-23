import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image
import numpy as np
import pdb
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F

class vae(nn.Module):
    def __init__(self, nz = 20):
        super(vae, self).__init__()
        self.nz = nz
        self.have_cuda = True
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # 8 x 8
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False), # 4 x 4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False), # 1 x 1
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False), # 4 x 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False), # 7 x 7
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 14 x 14
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 6, 2, 0, bias=False), # 32 x 32
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 32*32), x.view(-1, 32*32), size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + 3 * KLD, BCE, KLD

    def encode(self, x):
        conv = self.encoder(x);
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar
    
    def log(self, message):
        print(message)
        fh = open(path + "/train.log", 'a+')
        fh.write(message + '\n')
        fh.close()
        
        
    def save(self, model_path):
        self.log(f'Saving vae as {model_path}')
        model_state = {}
        model_state['vae'] = self.state_dict()
        model_state['optimizer'] = self.optimizer.state_dict()
        torch.save(model_state, model_path)
        
    def load(self, model_path):
        self.log(f'Loading saved vae named {model_path}') 
        model_state = torch.load(model_path)
        self.load_state_dict(model_state['vae'])
        self.optimizer.load_state_dict(model_state['optimizer'])
    
    def one_shot_prune(self, pruning_perc, layer_wise = False, trained_original_model_state = None):
        self.log(f"************pruning {pruning_perc} of the network****************")
        self.log(f"Layer-wise pruning = {layer_wise}")
        model = None
        if trained_original_model_state:
            model = torch.load(trained_original_model_state)
            
        if pruning_perc > 0:
            masks = {}
            flat_model_weights = np.array([])
            for name in model:
                #if "weight" in name:
                layer_weights = model[name].data.cpu().numpy()
                flat_model_weights = np.concatenate((flat_model_weights, layer_weights.flatten()))
            global_threshold = np.percentile(abs(flat_model_weights), pruning_perc)

            zeros = 0
            total = 0
            self.log("VAE layer-wise pruning percentages")
            for name in model:
                #if "weight" in name:
                weight_copy = model[name].data.abs().clone()
                threshold = global_threshold
                if layer_wise:
                    layer_weights = model[name].data.cpu().numpy()
                    threshold = np.percentile(abs(layer_weights), pruning_perc)
                mask = weight_copy.gt(threshold).int()
                self.log(f'{name} : {mask.numel() - mask.nonzero().size(0)} / {mask.numel()}  {(mask.numel() - mask.nonzero().size(0))/mask.numel()}')
                zeros += mask.numel() - mask.nonzero().size(0)
                total += mask.numel()
                masks[name] = mask
            self.log(f"Fraction of weights pruned = {zeros}/{total} = {zeros/total}")
            self.masks = masks
    
    def get_percent(self, total, percent):
        return (percent/100)*total


    def get_weight_fractions(self, number_of_iterations, percent):
        percents = []
        for i in range(number_of_iterations+1):
            percents.append(self.get_percent(100 - sum(percents), percent))
        self.log(f"{percents}")
        weight_fractions = []
        for i in range(1, number_of_iterations+1):
            weight_fractions.append(sum(percents[:i]))

        self.log(f"Weight fractions: {weight_fractions}")

        return weight_fractions
    
    def iterative_prune(self, init_state, 
                        trained_original_model_state, 
                        number_of_iterations, 
                        percent = 20, 
                        init_with_old = True):
        
        
        weight_fractions = self.get_weight_fractions(number_of_iterations, percent)       
        self.log("***************Iterative Pruning started. Number of iterations: {} *****************".format(number_of_iterations))
        for pruning_iter in range(0, number_of_iterations):
            self.log("Running pruning iteration {}".format(pruning_iter))
            self.__init__()
            self = self.cuda()
            trained_model = trained_original_model_state
            if pruning_iter != 0:
                trained_model = path + "/"+ "end_of_" + str(pruning_iter - 1) + '.pth'
            
            self.one_shot_prune(weight_fractions[pruning_iter], trained_original_model_state = trained_model)
            model.train(prune = True, init_state = init_state, init_with_old = init_with_old)
            torch.save(self.state_dict(), path + "/"+ "end_of_" + str(pruning_iter) + '.pth')
            
            sample, mu, logvar = self.forward(test_input.cuda())
            save_image(sample, path + '/image_' + str(pruning_iter) + '.png')

        self.log("Finished Iterative Pruning")
        
        
    def mask(self):
        model = self.state_dict()
        for name in model:
            #if "weight" in name:
            model[name].data.mul_(self.masks[name])
        self.load_state_dict(model)
    
    def train(self, prune, init_state, init_with_old):
        self.log(f"Number of parameters in model {sum(p.numel() for p in self.parameters())}")
        if not prune:
            self.save('./mnist/before_train.pth')
        
        if prune and init_with_old:
            self.load(init_state)
        
        for epoch in range(num_epochs):
            for data in dataloader:
                images, _ = data
                images = images.cuda()
                recon_images, mu, logvar = self.forward(images)
                loss, bce, kld = self.loss_fn(recon_images, images, mu, logvar)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if prune:
                    self.mask()

            self.log("Epoch[{}/{}] Loss: {} BCE: {} KL: {}".format(epoch+1, 
                                    num_epochs, loss.data.item()/batch_size, bce.data.item()/batch_size, kld.data.item()/batch_size))
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                #save_image(torch.cat((output, img), 0) * 0.5 + 0.5, './cifar/image_{}.png'.format(epoch))
                sample, mu, logvar = self.forward(test_input.cuda())
                save_image(sample, path + '/image_{}.png'.format(epoch))
        
        torch.save(self.state_dict(), path + '/vae.pth')

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

#dataset = CIFAR10('../datasets/cifar', download=True, transform=img_transform)
dataset = MNIST('../datasets/mnist', download=True, transform=img_transform)
dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

ngf = 128
ndf = 128
nc = 1
       
test_input, classes = next(iter(dataloader1)) 
print(classes)
    
path = './mnist_80'
init_state = './mnist/before_train.pth'
trained_original_model_state = './mnist/vae.pth'
init_with_old = True

fh = open(path + "/train.log", 'w')
fh.write('Logging')
fh.close()

model = vae().cuda()
model.one_shot_prune(80, trained_original_model_state = trained_original_model_state)
model.train(prune = True, init_state = init_state, init_with_old = init_with_old)
'''
model.iterative_prune(init_state = init_state, 
                    trained_original_model_state = trained_original_model_state, 
                    number_of_iterations = 20, 
                    percent = 20, 
                    init_with_old = init_with_old)

'''

