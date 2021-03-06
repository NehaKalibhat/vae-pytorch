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
from compute_flops import print_model_param_nums, print_model_param_flops

class EarlyBird():
    def __init__(self, percent, epoch_keep=5):
        self.percent = percent
        self.epoch_keep = epoch_keep
        self.masks = []
        self.dists = [1 for i in range(1, self.epoch_keep)]

    def pruning(self, model, percent):
        total = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total += m.weight.data.shape[0]

        bn = torch.zeros(total)
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)
        thre_index = int(total * percent)
        thre = y[thre_index]
        # print('Pruning threshold: {}'.format(thre))

        mask = torch.zeros(total)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.numel()
                weight_copy = m.weight.data.abs().clone()
                _mask = weight_copy.gt(thre.cuda()).float().cuda()
                mask[index:(index+size)] = _mask.view(-1)
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
                index += size

        # print('Pre-processing Successful!')
        return mask

    def put(self, mask):
        if len(self.masks) < self.epoch_keep:
            self.masks.append(mask)
        else:
            self.masks.pop(0)
            self.masks.append(mask)

    def cal_dist(self):
        if len(self.masks) == self.epoch_keep:
            for i in range(len(self.masks)-1):
                mask_i = self.masks[-1]
                mask_j = self.masks[i]
                self.dists[i] = 1 - float(torch.sum(mask_i==mask_j)) / mask_j.size(0)
            return True
        else:
            return False

    def early_bird_emerge(self, model):
        mask = self.pruning(model, self.percent)
        self.put(mask)
        flag = self.cal_dist()
        if flag == True:
            print(self.dists)
            for i in range(len(self.dists)):
                if self.dists[i] > 0.1:
                    return False
            return True
        else:
            return False
        
        
class vae(nn.Module):
    def __init__(self, input_dim, dim, z_dim = 128):
        super(vae, self).__init__()
        self.z_dim = z_dim
        self.device = 'cuda:0'
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 5, 1, 0),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
#             nn.BatchNorm2d(z_dim * 2)
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, dim, 5, 1, 0),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
#             nn.Tanh()
#         )
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim*2, 4, 2, 1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.Conv2d(dim*2, dim*4 , 4, 2, 1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            nn.Conv2d(dim*4, dim*8, 4, 2, 1),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),
            nn.Conv2d(dim*8, z_dim * 2, 2, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim*8, 2, 1, 0),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim*4, dim*2, 4, 2, 1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim*2, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )
        
#         self.encoder = nn.DataParallel(self.encoder)
#         self.decoder = nn.DataParallel(self.decoder)
            
        self.best_is = 0
        self.best_fid = 0
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

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
                mu = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                logvar = torch.randn(batch_size, self.z_dim, 1, 1, device=self.device)
                q_z_x = Normal(mu, logvar.mul(.5).exp())
                gen = self.decoder(q_z_x.rsample())
                gen = gen * 0.5 + 0.5
                gen = gen * 255.0
                gen = gen.cpu().numpy().astype(np.uint8)
                gen = np.transpose(gen, (0, 2, 3, 1))
                samples.extend(gen)
                
                torch.cuda.empty_cache()

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
                    torch.cuda.empty_cache()
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
            self.__init__(input_dim = nc, dim = hidden_size, z_dim = latent_size)
            self = self.to(self.device)
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
            
            sample, kl = self.forward(test_input.to(self.device))
            save_image(sample * 0.5 + 0.5, path + '/image_' + str(pruning_iter) + '.png')
            torch.cuda.empty_cache()
            
        self.log("Finished Iterative Pruning")
        
        
    def mask(self, prune_encoder = True, prune_decoder = True):
        model = self.state_dict()
        for name in model:
            if ("encoder" in name and prune_encoder) or ("decoder" in name and prune_decoder):
                model[name].data.mul_(self.masks[name])
        self.load_state_dict(model)
    
    def train(self, pruning_perc):
        self.log(f"Number of parameters in model {sum(p.numel() for p in self.parameters())}")
        
        self.log(f'original model param: {print_model_param_nums(self)}')
        self.log(f'original model flops: {print_model_param_flops(self, image_size, True)}')
        
        
        eb = EarlyBird(pruning_perc, epoch_keep = 5)
        found_eb = False
        
        for epoch in range(num_epochs):
            if not found_eb and eb.early_bird_emerge(self):
                found_eb = True
                print("FOUND EARLY BIRD TICKET, Pruning model to ", pruning_perc)
                self = one_shot_prune(self, pruning_perc)

                self.log(f'pruned model param: {print_model_param_nums(self)}')
                self.log(f'pruned model flops: {print_model_param_flops(self, image_size, True)}')

            for data in dataloader:
                x, _ = data
                x = x.to(self.device)
                
                x_tilde, kl_d = self.forward(x)
                loss_recons = F.mse_loss(x_tilde, x, size_average=False) / x.size(0)
                loss = loss_recons + 4 * kl_d

                nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
                log_px = nll.mean().item() - np.log(128) + kl_d.item()
                log_px /= np.log(2)

                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            self.log("Epoch[{}/{}] Loss: {} log_px:{} Recon: {} KL: {}".format(epoch+1, 
                                                                                      num_epochs, 
                                                                                      loss.data.item(), 
                                                                                      log_px,
                                                                                      loss_recons.data.item(), 
                                                                                      kl_d.data.item()))
            if epoch == num_epochs - 1:
                self.compute_inception_fid()
                sample, kl = self.forward(test_input.to(self.device))
                save_image(sample*0.5+0.5, path + '/image_{}.png'.format(pruning_perc))    
            
        torch.save(self.state_dict(), path + '/vae.pth')

def one_shot_prune(model, pruning_perc):
    
    for name, param in model.named_parameters():
        print(name, param.data.numel())
    
    
    model.log(f"************pruning {pruning_perc} of the network****************")

    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) and k != 15:
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    p_flops = 0
    y, i = torch.sort(bn)
    # comparsion and permutation (sort process)
    p_flops += total * np.log2(total) * 3
    thre_index = int(total * pruning_perc)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            if k == 15:
                mask = torch.ones_like(weight_copy).cuda()
            else:
                mask = weight_copy.gt(thre.cuda()).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            if int(torch.sum(mask)) > 0:
                cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    print(cfg)
    # apply mask flops
    # p_flops += total

    # compare two mask distance
    p_flops += 2 * total #(minus and sum)

    pruned_ratio = pruned/total
    print("Pruned ratio: ", pruned_ratio)
    print('  + Memory Request: %.2fKB' % float(total * 32 / 1024 / 8))
    print('  + Flops for pruning: %.2fM' % (p_flops / 1e6))

    newmodel = vae(input_dim = nc, dim = hidden_size, z_dim = latent_size)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        print("Layer id ", layer_id_in_cfg)
        if isinstance(m0, nn.BatchNorm2d):
            print(end_mask)
            if torch.sum(end_mask) == 0:
                continue
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = None
            if m1.weight.shape[0] == 1:
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.bias.data = m0.bias.data.clone()
                layer_id_in_cfg -= 1
            else:
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                
            m1.weight.data = w1.clone()
            
        elif isinstance(m0, nn.ConvTranspose2d):
            if torch.sum(end_mask) == 0:
                continue
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # random set for test
            # new_end_mask = np.asarray(end_mask.cpu().numpy())
            # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
            # idx1 = np.squeeze(np.argwhere(new_end_mask))

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = None
            if m1.weight.shape[0] == latent_size:
                w1 = m0.weight.data[:, idx1.tolist(), :, :].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                layer_id_in_cfg += 1
            elif m1.weight.shape[1] == 3:
                w1 = m0.weight.data[idx0.tolist(), :, :, :].clone()
                m1.bias.data = m0.bias.data.clone()
            else:   
                w1 = m0.weight.data[:, idx1.tolist(), :, :].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                w1 = w1[idx0.tolist(), :, :, :].clone()   
            
            m1.weight.data = w1.clone()

    #torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

    model = newmodel
    
    for name, param in model.named_parameters():
        print(name, param.data.numel())
        
    return model


# batch_size = 128
# img_transform = transforms.Compose([
#                 transforms.Resize(32),
#                 transforms.RandomCrop(32),
#                 #transforms.Grayscale(3),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#             ])

# dataset = datasets.ImageFolder("../datasets/celeba_short", transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
# dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# test_input, classes = next(iter(dataloader1))

# inception_tf.initialize_inception()

# inception_cache_path = './inception_cache/celeba'
# path = 'celeba_test'
# num_epochs = 100

# for i in range(19):
#     gan = torch.load("../gan-pytorch/celeba_iter_1/end_of_"+str(i)+".pth")
#     vae = vae(input_dim = 3, dim = 64, z_dim = 32)
#     vae = vae.to(vae.device)
#     model = vae.state_dict()
#     for name in gan:
#         if 'D.' in name and '9' not in name:
#             vae_layer = name.replace('D', 'encoder')
#         if 'G.' in name and '3' in name:
#             vae_layer = name.replace('G', 'decoder')
#             vae_layer = vae_layer.replace('3', '6')
#         if 'G.' in name and '4' in name:
#             vae_layer = name.replace('G', 'decoder')
#             vae_layer = vae_layer.replace('4', '7')
#         if 'G.' in name and '6' in name:
#             vae_layer = name.replace('G', 'decoder')
#             vae_layer = vae_layer.replace('6', '9')
#         if 'G.' in name and '7' in name:
#             vae_layer = name.replace('G', 'decoder')
#             vae_layer = vae_layer.replace('7', '10')
#         if 'G.' in name and '9' in name:
#             vae_layer = name.replace('G', 'decoder')
#             vae_layer = vae_layer.replace('9', '12')  

#         model[vae_layer] = gan[name]

#     vae.load_state_dict(model)
#     vae.train(prune = False)

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
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

    if 'cifar' in dataset:
        dataset = CIFAR10(dataset, download=True, transform=img_transform)
    if 'mnist' in dataset:
        dataset = MNIST(dataset, download=True, transform=img_transform)
    if 'celeba' in dataset:
        dataset = datasets.ImageFolder(dataset, transform=img_transform)

    dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    nc = 3

    test_input, classes = next(iter(dataloader1))
    print(classes)


    fh = open(path + "/train.log", 'w')
    fh.write('Logging')
    fh.close()

    inception_tf.initialize_inception()

    model = vae(input_dim = nc, dim = hidden_size, z_dim = latent_size)
    model = model.to(model.device)
    #model.one_shot_prune(80, trained_original_model_state = trained_original_model_state)
    model.train(0.3)

#     model.iterative_prune(init_state = init_state, 
#                         trained_original_model_state = trained_original_model_state, 
#                         number_of_iterations = 20, 
#                         percent = 20, 
#                         init_with_old = init_with_old,
#                         prune_encoder = prune_encoder,
#                         prune_decoder = prune_decoder)

