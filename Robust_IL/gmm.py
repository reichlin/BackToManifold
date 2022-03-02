import os
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from models import Equivariant_map_images, Mixture_Density_Network
from torch.utils.tensorboard import SummaryWriter

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Estimator():

    def __init__(self, transform_type, device, task="pick"):
        super(Estimator, self).__init__()

        self.task = task

        self.device = device
        self.d = 3
        channels = 3
        self.z_dims = self.d
        self.N = 10

        self.E = Equivariant_map_images(channels, self.d, transform_type).to(self.device)
        if os.path.isdir('../files/models/equivariance'):
            self.load_E("../files/models/equivariance/best_E_model.mdl")
        else:
            print("ERROR - Equivariant model not trained yet")
            exit()
        self.E.eval()

        self.GMM = Mixture_Density_Network(self.N, self.d, transform_type).to(self.device)

        self.gmm_folder = '../files/models/pick/GMM_model' if task == "pick" else '../files/models/push/GMM_model'
        if os.path.isdir(self.gmm_folder):
            self.load_gmm(self.gmm_folder + '/best_GMM_model.mdl')
        self.GMM.eval()

        return

    def get_probability_image(self, o0, ot, gt):

        zt = self.E(ot)

        mu, sigma, w, _ = self.GMM(o0.repeat(gt.shape[0], 1, 1, 1), gt)
        comp = MultivariateNormal(mu, torch.diag_embed(sigma))
        mix = Categorical(w)
        gmm = MixtureSameFamily(mix, comp)
        likelihood = gmm.log_prob(zt)

        return likelihood, zt

    def get_probability_z(self, o0, z, g):

        mu, sigma, w, _ = self.GMM(o0.repeat(g.shape[0], 1, 1, 1), g)
        comp = MultivariateNormal(mu, torch.diag_embed(sigma))
        mix = Categorical(w)
        gmm = MixtureSameFamily(mix, comp)
        likelihood = gmm.log_prob(z)

        return likelihood

    def get_recovery_action(self, o0, ot, gt):

        z = self.E(ot)

        p = self.get_probability_z(o0, z, gt)
        grad = torch.autograd.grad(outputs=p, inputs=z)[0]

        if torch.any(torch.isnan(grad)):
            grad = torch.nan_to_num(grad, nan=0.)

        a = np.zeros(4)
        a[0:3] = (grad[0]).detach().cpu().numpy()

        return p, z, a

    def fit_gmm(self, dataloader, save_model=False):

        best_nll = 1000
        EPOCHS = 100000
        batch_size = 64

        optimizer_gmm = optim.Adam(self.GMM.parameters(), lr=1e-4)
        writer = SummaryWriter("../logs/logs_gmm/dataset4_correct_n10_restr_dec3") if self.task == "pick" else SummaryWriter("../logs/logs_gmm/2")
        log_idx_gmm_train = 0

        for epoch in tqdm(range(EPOCHS)):

            self.GMM.train()

            # TRAINING
            avg_nll_train = 0
            n_steps = int(dataloader.len_train / batch_size)+1
            for _ in range(n_steps):

                ot, o0, g = dataloader.get_batch_gmm(batch_size, 'train')
                ot, o0, g = ot.to(self.device), o0.to(self.device), g.view(-1, 1).to(self.device)

                zt = self.E(ot).detach()

                mu, sigma, w, o0_hat = self.GMM(o0, g)

                comp = MultivariateNormal(mu, torch.diag_embed(sigma))
                mix = Categorical(w)
                gmm = MixtureSameFamily(mix, comp)
                likelihood = gmm.log_prob(zt)
                loss = -likelihood.mean()

                loss += torch.mean((o0 - o0_hat) ** 2)

                optimizer_gmm.zero_grad()
                loss.backward()
                optimizer_gmm.step()

                avg_nll_train += loss.detach().cpu().item()

            avg_nll_train /= n_steps

            # EVAL
            self.GMM.eval()
            avg_nll_test = 0
            n_steps = int(dataloader.len_test / batch_size) + 1
            for _ in range(n_steps):
                ot, o0, g = dataloader.get_batch_gmm(batch_size, 'test')
                ot, o0, g = ot.to(self.device), o0.to(self.device), g.view(-1, 1).to(self.device)

                zt = self.E(ot).detach()

                mu, sigma, w, _ = self.GMM(o0, g)

                comp = MultivariateNormal(mu, torch.diag_embed(sigma))
                mix = Categorical(w)
                gmm = MixtureSameFamily(mix, comp)
                likelihood = gmm.log_prob(zt)
                loss = -likelihood.mean()

                optimizer_gmm.zero_grad()
                loss.backward()
                optimizer_gmm.step()

                avg_nll_test += loss.detach().cpu().item()

            avg_nll_test /= n_steps

            if epoch % 100 == 99:
                fig = plt.figure()
                for n in range(self.N):
                    plt.scatter(mu[:, n, 1].detach().cpu(), -mu[:, n, 0].detach().cpu())
                writer.add_figure("Mu", fig, log_idx_gmm_train)

            writer.add_scalar("NLL_GMM_TRAIN", avg_nll_train, log_idx_gmm_train)
            writer.add_scalar("NLL_GMM_TEST", avg_nll_test, log_idx_gmm_train)
            log_idx_gmm_train += 1

            # if (save_model and best_nll > avg_nll or epoch) or (save_model and epoch % 1 == 0):
            if save_model and best_nll > avg_nll_test:
                best_nll = avg_nll_test
                if not os.path.isdir(self.gmm_folder):
                    os.makedirs(self.gmm_folder)
                fname = self.gmm_folder+"/epoch="+str(log_idx_gmm_train)+"_nll="+str(avg_nll_test)+".mdl"
                torch.save(self.GMM.state_dict(), fname)

    def load_E(self, fname):
        state_dict = torch.load(fname, map_location=self.device)
        self.E.load_state_dict(state_dict)
        self.E.eval()

    def load_gmm(self, fname):
        state_dict = torch.load(fname, map_location=self.device)
        self.GMM.load_state_dict(state_dict)
        self.GMM.eval()













