import os
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from models import Equivariant_map_images
from torch.utils.tensorboard import SummaryWriter

# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt


class E_encoder():

    def __init__(self, transform_type, device, task="pick"):
        super(E_encoder, self).__init__()

        self.task = task

        self.device = device
        self.d = 3
        channels = 3
        self.z_dims = self.d

        self.E = Equivariant_map_images(channels, self.d, transform_type).to(self.device)
        if os.path.isdir('../files/models/equivariance'):
            self.load_E("../files/models/equivariance/best_E_model.mdl")
        self.E.eval()

        z_folder = '../files/models/pick/z_dataset' if task == "pick" else '../files/models/push/z_dataset'
        if os.path.isdir(z_folder):
            self.z = torch.from_numpy(np.load(z_folder+'/z.npy')).to(self.device)

        return

    def generate_equivariant_trajectories(self, dataloader):

        # zx = np.linspace(-0.1, 1.1, 120)
        # zy = np.linspace(-0.1, 1.6, 170)
        # zz = np.linspace(-0.3, 0.3, 10)
        # density_pre = np.zeros((120, 170, 10))
        # density_post = np.zeros((120, 170, 10))
        #
        # # dataloader.trj_ctr = 0
        # # imgs, actions, states = dataloader.get_trajectory()
        # # imgs = imgs.to(self.device)
        # # plt.figure()
        # # plt.imshow(torch.transpose(torch.transpose(imgs[0], 1, 0), 2, 1).detach().cpu())
        # # plt.show()
        #
        # imgs, actions, states = dataloader.get_trajectory()
        # imgs = imgs.to(self.device)
        # grasp_pre = torch.zeros(1, 1).to(self.device)
        # grasp_post = torch.ones(1, 1).to(self.device)
        #
        # mu_pre, sigma_pre, w_pre, _ = self.GMM(imgs[0:1].repeat(grasp_pre.shape[0], 1, 1, 1), grasp_pre)
        # comp_pre = MultivariateNormal(mu_pre, torch.diag_embed(sigma_pre))
        # mix_pre = Categorical(w_pre)
        # gmm_pre = MixtureSameFamily(mix_pre, comp_pre)
        #
        # mu_post, sigma_post, w_post, _ = self.GMM(imgs[0:1].repeat(grasp_post.shape[0], 1, 1, 1), grasp_post)
        # comp_post = MultivariateNormal(mu_post, torch.diag_embed(sigma_post))
        # mix_post = Categorical(w_post)
        # gmm_post = MixtureSameFamily(mix_post, comp_post)
        #
        # with torch.no_grad():
        #     for i in tqdm(range(120)):
        #         for j in range(170):
        #             for k in range(10):
        #                 z = torch.zeros(1, 3).to(self.device)
        #                 z[0, 0] = zx[i]
        #                 z[0, 1] = zy[j]
        #                 z[0, 2] = zz[k]
        #                 density_pre[i, j, k] = gmm_pre.log_prob(z).cpu().item()
        #                 density_post[i, j, k] = gmm_post.log_prob(z).cpu().item()
        #
        # print()

        print("Generate Datapoints:")
        with torch.no_grad():
            # plt.figure()
            self.z = None
            for _ in tqdm(range(dataloader.n_trj)):
                imgs, actions, states = dataloader.get_trajectory()
                imgs, actions, states = imgs.to(self.device), actions.to(self.device), states.to(self.device)
                z = self.E(imgs)
                # plt.scatter(z[0, 1].detach().cpu().numpy(), -z[0, 0].detach().cpu().numpy(), color='red')
                # plt.scatter(z[1:, 1].detach().cpu().numpy(), -z[1:, 0].detach().cpu().numpy(), color='blue')
                # plt.plot(z[:, 1].detach().cpu().numpy(), -z[:, 0].detach().cpu().numpy())
                if self.z is None:
                    self.z = z
                else:
                    self.z = torch.cat([self.z, z], 0)
            # plt.show()
            # print()

        folder = '../files/models/pick/z_dataset' if self.task == 'pick' else '../files/models/push/z_dataset'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        fname_z = folder + "/z.npy"
        np.save(fname_z, self.z.detach().cpu().numpy())

    def fit_E(self, dataloader, save_model=False):

        precision_min = 100
        batch_size = 64
        EPOCHS = 10000

        optimizer_E = optim.Adam(self.E.parameters(), lr=1e-3)
        writer = SummaryWriter("../logs/logs_e/dataset_correct")
        log_idx_e_train = 0
        log_idx_e_test = 0

        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        for epoch in tqdm(range(EPOCHS)):

            self.E.train()

            # TRAINING
            avg_equivariance_loss = 0
            avg_cos_sim = 0
            avg_origin_loss = 0
            n_steps = int(dataloader.len_train / batch_size) + 1
            for _ in range(n_steps):
                ot, ot1, at, s0 = dataloader.get_batch(batch_size, 'train')
                ot, ot1, at, s0 = ot.to(self.device), ot1.to(self.device), at.to(self.device), s0.to(self.device)

                zt = self.E(ot)
                zt1 = self.E(ot1)
                z0 = self.E(s0)

                loss_o = torch.mean(torch.sum((z0) ** 2, -1))
                loss_E = torch.mean(torch.sum(((zt1 - zt) - at) ** 2, -1))

                loss = loss_E + loss_o

                optimizer_E.zero_grad()
                loss.backward()
                optimizer_E.step()

                avg_equivariance_loss += loss_E.detach().cpu().item()
                avg_cos_sim += torch.sum(cos_sim((zt1 - zt), at)).detach().cpu().item()
                avg_origin_loss += loss_o.detach().cpu().item()

            avg_equivariance_loss /= n_steps
            avg_cos_sim /= (n_steps * batch_size)
            avg_origin_loss /= n_steps

            writer.add_scalar("Train/avg_equivariance_loss", avg_equivariance_loss, log_idx_e_train)
            writer.add_scalar("Train/avg_cos_sim", avg_cos_sim, log_idx_e_train)
            writer.add_scalar("Train/avg_origin_loss", avg_origin_loss, log_idx_e_train)
            log_idx_e_train += 1

            self.E.eval()

            # TESTING
            avg_precision = 0
            avg_equivariance_loss = 0
            avg_cos_sim = 0
            n_steps = int(dataloader.len_train / batch_size) + 1
            for step in range(n_steps):
                ot, ot1, at, _ = dataloader.get_batch(batch_size, 'test')
                ot, ot1, at = ot.to(self.device), ot1.to(self.device), at.to(self.device)

                zt = self.E(ot)
                zt1 = self.E(ot1)

                avg_precision += torch.sum(torch.mean(torch.abs((zt1 - zt) - at), -1)).detach().cpu().item()
                avg_equivariance_loss += torch.sum(((zt1 - zt) - at) ** 2).detach().cpu().item()
                avg_cos_sim += torch.sum(cos_sim((zt1 - zt), at)).detach().cpu().item()

            avg_precision /= (n_steps * batch_size)
            avg_equivariance_loss /= (n_steps * batch_size)
            avg_cos_sim /= (n_steps * batch_size)

            writer.add_scalar("Test/avg_precision", avg_precision, log_idx_e_test)
            writer.add_scalar("Test/avg_equivariance_loss", avg_equivariance_loss, log_idx_e_test)
            writer.add_scalar("Test/avg_cos_sim", avg_cos_sim, log_idx_e_test)
            log_idx_e_test += 1

            if save_model and avg_precision < precision_min:
                precision_min = avg_precision
                folder = '../files/models/equivariance'
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                fname = folder + "/epoch=" + str(log_idx_e_test) + "_precision=" + str(avg_precision) + ".mdl"
                torch.save(self.E.state_dict(), fname)

    def load_E(self, fname):
        state_dict = torch.load(fname, map_location=self.device)
        self.E.load_state_dict(state_dict)
        self.E.eval()
















