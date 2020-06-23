import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image
import numpy as np
import pdb
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import inception_tf
import fid
import os.path as osp

class vae(nn.Module):
    def __init__(self, input_dim=784, dim=256, z_dim=8):
        super(vae, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 32), 
            nn.ReLU(True), 
            nn.Linear(32, self.z_dim*2))
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh())
        
        self.best_is = 0
        self.best_fid = 0
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x): 
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div
    
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
                mu = torch.randn(batch_size, self.z_dim).cuda()
                logvar = torch.randn(batch_size, self.z_dim).cuda()
                q_z_x = Normal(mu, logvar.mul(.5).exp())
                gen = self.decoder(q_z_x.rsample())
                gen = gen.view(-1, nc, image_size, image_size)
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
        
    def load(self, model_path, prune_encoder = True, prune_decoder = True):
        self.log(f'Loading saved vae named {model_path}') 
        model_state = torch.load(model_path)
        new_state = {}
        for name in model_state['vae']:
            if ("encoder" in name and prune_encoder) or ("decoder" in name and prune_decoder):
                self.log(f"loading {name}")
                new_state[name] = model_state['vae'][name]
            else:
                new_state[name] = self.state_dict()[name]
        self.load_state_dict(new_state)
        self.optimizer.load_state_dict(model_state['optimizer'])
    
    def one_shot_prune(self, pruning_perc, prune_encoder = True, prune_decoder = True, layer_wise = False, trained_original_model_state = None):
        self.log(f"************pruning {pruning_perc} of the network****************")
        self.log(f"Layer-wise pruning = {layer_wise}")
        model = None
        if trained_original_model_state:
            model = torch.load(trained_original_model_state)
        
        if pruning_perc > 0:
            masks = {}
            flat_model_weights = np.array([])
            for name in model:
                if ("encoder" in name and prune_encoder) or ("decoder" in name and prune_decoder):
                    layer_weights = model[name].data.cpu().numpy()
                    flat_model_weights = np.concatenate((flat_model_weights, layer_weights.flatten()))
            global_threshold = np.percentile(abs(flat_model_weights), pruning_perc)

            zeros = 0
            total = 0
            self.log("VAE layer-wise pruning percentages")
            for name in model:
                if ("encoder" in name and prune_encoder) or ("decoder" in name and prune_decoder):
                    threshold = global_threshold
                    if layer_wise:
                        layer_weights = model[name].data.cpu().numpy()
                        threshold = np.percentile(abs(layer_weights), pruning_perc)
                    masks[name] = model[name].abs().gt(threshold).int()
                    pruned = masks[name].numel() - masks[name].nonzero().size(0)
                    tot = masks[name].numel()
                    frac = pruned / tot
                    self.log(f"{name} : {pruned} / {tot}  {frac}")
                    zeros += pruned
                    total += tot
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
                        init_with_old = True,
                        prune_encoder = True,
                        prune_decoder = True):
        
        
        weight_fractions = self.get_weight_fractions(number_of_iterations, percent)       
        self.log("***************Iterative Pruning started. Number of iterations: {} *****************".format(number_of_iterations))
        for pruning_iter in range(0, number_of_iterations):
            self.log("Running pruning iteration {}".format(pruning_iter))
            self.__init__(input_dim = nc * image_size * image_size)
            self = self.cuda()
            trained_model = trained_original_model_state
            if pruning_iter != 0:
                trained_model = path + "/"+ "end_of_" + str(pruning_iter - 1) + '.pth'
            
            self.one_shot_prune(weight_fractions[pruning_iter], 
                                trained_original_model_state = trained_model, 
                                prune_encoder = prune_encoder,
                                prune_decoder = prune_decoder)
            self.train(prune = True, 
                       init_state = init_state, 
                       init_with_old = init_with_old,
                       prune_encoder = prune_encoder,
                       prune_decoder = prune_decoder)
            
            torch.save(self.state_dict(), path + "/"+ "end_of_" + str(pruning_iter) + '.pth')
            
            sample, kl = self.forward(test_input.cuda())
            sample = sample.view(-1, nc, image_size, image_size)
            save_image(sample * 0.5 + 0.5, path + '/image_' + str(pruning_iter) + '.png')
            torch.cuda.empty_cache()
            
        self.log("Finished Iterative Pruning")
        
        
    def mask(self, prune_encoder = True, prune_decoder = True):
        model = self.state_dict()
        for name in model:
            if ("encoder" in name and prune_encoder) or ("decoder" in name and prune_decoder):
                model[name].data.mul_(self.masks[name])
        self.load_state_dict(model)
    
    def train(self, prune, init_state, init_with_old, prune_encoder = True, prune_decoder = True):
        self.log(f"Number of parameters in model {sum(p.numel() for p in self.parameters())}")
        if not prune:
            self.save(path + '/before_train.pth')
        
        if prune and init_with_old:
            self.load(init_state, 
                      prune_encoder = prune_encoder,
                      prune_decoder = prune_decoder)
        
        for epoch in range(num_epochs):
            for data in dataloader:
                x, _ = data
                x = x.view(x.size(0), -1).cuda()
                
                x_tilde, kl_d = self.forward(x)
                loss_recons = F.mse_loss(x_tilde, x, size_average=False) / x.size(0)
                loss = loss_recons + kl_d

                nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
                log_px = nll.mean().item() - np.log(128) + kl_d.item()
                log_px /= np.log(2)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if prune:
                    self.mask(prune_encoder = prune_encoder,
                              prune_decoder = prune_decoder)

            self.log("Epoch[{}/{}] Loss: {} log_px:{} Recon: {} KL: {}".format(epoch+1, 
                                                                                      num_epochs, 
                                                                                      loss.data.item(), 
                                                                                      log_px,
                                                                                      loss_recons.data.item(), 
                                                                                      kl_d.data.item()))
#             if epoch > 0 and (epoch % 20 == 0 or epoch == num_epochs - 1):
#                 #self.compute_inception_fid()
#                 sample, kl = self.forward(test_input.cuda())
#                 sample = sample.view(-1, nc, image_size, image_size)
#                 save_image(sample * 0.5 + 0.5, path + '/image_{}.png'.format(epoch))    
            
        torch.save(self.state_dict(), path + '/vae.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch VAE')

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--latent_size', default=32, type=int)
    parser.add_argument('--dataset', default='../datasets/cifar', type=str)
    parser.add_argument('--inception_cache_path', default='./inception_cache/cifar', type=str)
    parser.add_argument('--log_path', default='./cifar_test', type=str)
    parser.add_argument('--init_state', default='./cifar/before_train.pth', type=str)
    parser.add_argument('--trained_original_model_state', default='./cifar/vae.pth', type=str)
    parser.add_argument('--init_with_old', default='True', type=str)
    parser.add_argument('--prune_encoder', default='True', type=str)
    parser.add_argument('--prune_decoder', default='True', type=str)

    args = parser.parse_args()

    if args.num_epochs != '':
         num_epochs = args.num_epochs

    if args.batch_size != '':
         batch_size = args.batch_size

    if args.lr != '':
         learning_rate = args.lr

    if args.image_size != '':
         image_size = args.image_size

    if args.hidden_size != '':
         hidden_size = args.hidden_size

    if args.latent_size != '':
         latent_size = args.latent_size

    if args.dataset != '':
         dataset = args.dataset

    if args.inception_cache_path != '':
         inception_cache_path = args.inception_cache_path

    if args.log_path != '':
         path = args.log_path

    if args.init_state != '':
         init_state = args.init_state

    if args.trained_original_model_state != '':
         trained_original_model_state = args.trained_original_model_state

    if args.init_with_old != '':
         init_with_old = args.init_with_old == 'True'

    if args.prune_encoder != '':
         prune_encoder = args.prune_encoder == 'True'

    if args.prune_decoder != '':
         prune_decoder = args.prune_decoder == 'True'

    img_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                #transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    if 'cifar' in dataset:
        dataset = CIFAR10(dataset, download=True, transform=img_transform)
    if 'mnist' in dataset:
        dataset = MNIST(dataset, download=True, transform=img_transform)
    if 'celeba' in dataset:
        dataset = datasets.ImageFolder(dataset, transform=img_transform)

    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    nc = 1

    test_input, classes = next(iter(dataloader1)) 
    test_input = test_input.view(test_input.size(0), -1)
    print(classes)


    fh = open(path + "/train.log", 'w')
    fh.write('Logging')
    fh.close()

    inception_tf.initialize_inception()

    model = vae(input_dim = nc * image_size * image_size).cuda()
    #model.one_shot_prune(80, trained_original_model_state = trained_original_model_state)
    #model.train(prune = False, init_state = init_state, init_with_old = init_with_old)

    model.iterative_prune(init_state = init_state, 
                        trained_original_model_state = trained_original_model_state, 
                        number_of_iterations = 20, 
                        percent = 20, 
                        init_with_old = init_with_old,
                        prune_encoder = prune_encoder,
                        prune_decoder = prune_decoder)

