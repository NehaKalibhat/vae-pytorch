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
import inception_tf
import fid
import os.path as osp

class vae(nn.Module):
    def __init__(self, nz = 128):
        super(vae, self).__init__()
        self.nz = nz
        self.have_cuda = True
        self.device = 'cuda'
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ngf, kernel_size=2, stride=2), # 16x16
            nn.ReLU(),
            nn.Conv2d(ngf, ngf*2, kernel_size=2, stride=2), # 8x8
            nn.ReLU(),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=2, stride=2), # 4x4
            nn.ReLU(),
            nn.Conv2d(ngf*4, ngf*8, kernel_size=2, stride=2), # 2x2
            nn.ReLU(),
            nn.Conv2d(ngf*8, ngf*16, kernel_size=2, stride=2), # 1x1
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ndf*16, ndf*8, kernel_size=2, stride=2), # 2x2
            nn.ReLU(),
            nn.ConvTranspose2d(ndf*8, ndf*4, kernel_size=2, stride=2), # 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(ndf*4, ndf*2, kernel_size=2, stride=2), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(ndf*2, ndf, kernel_size=2, stride=2), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(ndf, nc, kernel_size=2, stride=2), # 32x32
            nn.Tanh(),
        )

        self.fc1 = nn.Linear(1024, nz)
        self.fc2 = nn.Linear(1024, nz)
        self.fc3 = nn.Linear(nz, 1024)
        
        self.best_is = 0
        self.best_fid = 0
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = nn.MSELoss(size_average=False)(recon_x, x)/x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/x.size(0)
        return BCE + KLD, BCE, KLD

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp.cuda()
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z.view(-1,1024,1,1))
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
    
    def log(self, message):
        print(message)
        fh = open(path + "/train.log", 'a+')
        fh.write(message + '\n')
        fh.close()
        
    def compute_inception_score(self, samples):
        IS_mean, IS_std = inception_tf.get_inception_score(np.array(samples), splits=10,
                                                           batch_size=100, mem_fraction=1)
        print('Inception score: {} +/- {}'.format(IS_mean, IS_std))
        return IS_mean, IS_std

    def compute_fid(self, samples):
        fid_score = fid.compute_fid(osp.join(inception_cache_path, 'stats.npy'), samples,
                                    inception_cache_path, dataloader)
        print('FID score: {}'.format(fid_score))
        return fid_score
    
    def compute_inception_fid(self):
        samples = []
        num_batches = int(50000 / batch_size)
        for batch in range(num_batches):
            with torch.no_grad():
                z = torch.randn(batch_size, self.nz, device=self.device)
                gen = self.decode(z)
                gen = gen * 0.5 + 0.5
                gen = gen * 255.0
                gen = gen.cpu().numpy().astype(np.uint8)
                gen = np.transpose(gen, (0, 2, 3, 1))
                samples.extend(gen)

        IS_mean, IS_std = self.compute_inception_score(samples)
        fid = self.compute_fid(samples)
        self.log('IS: {} +/- {}'.format(IS_mean, IS_std))
        self.log('FID: {}'.format(fid))
        if self.best_is < IS_mean:
            self.best_is = IS_mean
            self.best_is_std = IS_std
            self.best_fid = fid
        if self.best_fid > fid:
            self.best_fid = fid
        self.log('Best IS: {} +/- {}'.format(self.best_is, self.best_is_std))
        self.log('Best FID: {}'.format(self.best_fid))    
            
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
            self.save('./cifar/before_train.pth')
        
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

            self.log("Epoch[{}/{}] Loss: {} Recon: {} KL: {}".format(epoch+1, 
                                    num_epochs, loss.data.item(), bce.data.item(), kld.data.item()))
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.compute_inception_fid()
                #save_image(torch.cat((output, img), 0) * 0.5 + 0.5, './cifar/image_{}.png'.format(epoch))
                sample, mu, logvar = self.forward(test_input.cuda())
                save_image(sample*0.5+0.5, path + '/image_{}.png'.format(epoch))
    
        torch.save(self.state_dict(), path + '/vae.pth')

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

dataset = CIFAR10('../datasets/cifar', download=True, transform=img_transform)
#dataset = MNIST('../datasets/mnist', download=True, transform=img_transform)
dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

ngf = 64
ndf = 64
nc = 3
       
test_input, classes = next(iter(dataloader1)) 
print(classes)

inception_cache_path = './inception_cache/cifar'
path = './cifar'
init_state = './cifar/before_train.pth'
trained_original_model_state = './cifar/vae.pth'
init_with_old = True

fh = open(path + "/train.log", 'w')
fh.write('Logging')
fh.close()

inception_tf.initialize_inception()

model = vae().cuda()
#model.one_shot_prune(80, trained_original_model_state = trained_original_model_state)
model.train(prune = False, init_state = init_state, init_with_old = init_with_old)
'''
model.iterative_prune(init_state = init_state, 
                    trained_original_model_state = trained_original_model_state, 
                    number_of_iterations = 20, 
                    percent = 20, 
                    init_with_old = init_with_old)

'''

