import torch
import torchvision.transforms as transforms
import PIL
import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm

# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt


class EDataLoader:
    def __init__(self, dataset_dir, transform_type=0, apply_normalization=False, load_imgs=False):
        self.apply_normalization = apply_normalization

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        if transform_type == 0:
            self.image_size_w = 256
            self.image_size_h = 256
            self.transform = transforms.Compose([transforms.CenterCrop(480),
                                                 transforms.Resize((self.image_size_h, self.image_size_w)),
                                                 transforms.ToTensor()])
        elif transform_type == 1:
            self.image_size_w = 256
            self.image_size_h = 192
            self.transform = transforms.Compose([transforms.CenterCrop((480, 640)),
                                                 transforms.Resize((self.image_size_h, self.image_size_w)),
                                                 transforms.ToTensor()])
        elif transform_type == 2:
            self.image_size_w = 224
            self.image_size_h = 224
            self.transform = transforms.Compose([transforms.CenterCrop(640),
                                                 transforms.Resize((self.image_size_h, self.image_size_w)),
                                                 transforms.ToTensor()])
        self.dir = dataset_dir
        self.load_imgs = load_imgs
        folders = os.listdir(dataset_dir)
        self.actions = None
        self.imgs = None
        self.groups = None
        self.init_states = []

        self.max_T = 3

        print("Load Data in RAM:")
        group = 0
        self.folders_name = {}
        n_counter = 0

        # Load init images
        self.init_imgs = torch.zeros(len(folders), 3, self.image_size_h, self.image_size_w)
        for j, folder in enumerate(folders):
            self.init_imgs[j] = self.process_img(PIL.Image.open(self.dir + "/" + folder + '/img{:04d}.png'.format(0)))

        # Load train data
        for folder in tqdm(folders):

            actions = torch.from_numpy(np.load(self.dir + "/" + folder + '/actions.npy')).float()
            actions = actions[:, :3]
            n_data = actions.shape[0]

            if self.load_imgs:
                imgs = torch.zeros(n_data, 3, self.image_size_h, self.image_size_w)
                for i in range(n_data):
                    imgs[i] = self.process_img(PIL.Image.open(self.dir + "/" + folder + '/img{:04d}.png'.format(i)))
            else:
                for i in range(n_data):
                    self.folders_name[n_counter] = self.dir + "/" + folder + '/img{:04d}.png'.format(i)
                    n_counter += 1

            if self.actions is None:
                self.actions = actions
                self.groups = torch.ones(actions.shape[0]) * group
                if self.load_imgs:
                    self.imgs = imgs

            else:
                self.actions = torch.cat([self.actions, actions], 0)
                self.groups = torch.cat([self.groups, torch.ones(actions.shape[0]) * group], 0)
                if self.load_imgs:
                    self.imgs = torch.cat([self.imgs, imgs], 0)

            group += 1

        tot_len = self.actions.shape[0] - self.max_T
        idx = np.arange(tot_len)
        np.random.shuffle(idx)
        self.len_train = int(tot_len*0.8)
        self.len_test = tot_len - self.len_train
        self.train_idx = idx[:self.len_train]
        self.test_idx = idx[self.len_train:]

        self.train_ctr = 0
        self.test_ctr = 0

    def get_batch(self, batch_size, source):

        if source == 'train':
            idx = self.train_idx[self.train_ctr:min(self.train_ctr + batch_size, self.len_train-1)]
            if (self.train_ctr + batch_size) >= self.len_train:
                np.random.shuffle(self.train_idx)
                idx = np.append(idx, self.train_idx[0:(self.train_ctr + batch_size - self.len_train)])
                self.train_ctr = (self.train_ctr + batch_size - self.len_train)
            else:
                self.train_ctr += batch_size

        elif source == 'test':
            idx = self.test_idx[self.test_ctr:min(self.test_ctr + batch_size, self.len_test-1)]
            if (self.test_ctr + batch_size) >= self.len_test:
                np.random.shuffle(self.test_idx)
                idx = np.append(idx, self.test_idx[0:(self.test_ctr + batch_size - self.len_test)])
                self.test_ctr = (self.test_ctr + batch_size - self.len_test)
            else:
                self.test_ctr += batch_size

        T = np.random.randint(1, self.max_T + 1, idx.shape)
        coherent = (self.groups[idx] == self.groups[idx+T]) * 1.
        idx = idx[np.nonzero(coherent)[:,0]]
        T = T[np.nonzero(coherent)[:,0]]

        if self.load_imgs:
            ot = self.imgs[idx]
            ot1 = self.imgs[idx + T]
        else:
            ot = torch.zeros(idx.shape[0], 3, self.image_size_h, self.image_size_w)
            ot1 = torch.zeros(idx.shape[0], 3, self.image_size_h, self.image_size_w)
            for i, j in enumerate(idx):
                ot[i] = self.process_img(PIL.Image.open(self.folders_name[int(j)]))
                ot1[i] = self.process_img(PIL.Image.open(self.folders_name[int(j+T[i])]))

        at = torch.cat([torch.sum(self.actions[i:i + t], 0).view(1, 3) for i, t in zip(idx, T)], 0)

        if source == 'train':
            s0 = self.init_imgs[self.groups[idx].long()]
        else:
            s0 = None

        return ot, ot1, at, s0

    def process_img(self, img):
        if self.apply_normalization:
            return self.normalize(self.transform(img))
        return self.transform(img)


if __name__ == '__main__':

    dataloader = EDataLoader('../files/data/equivariance/dataset_equivariance', transform_type=2, apply_normalization=True, load_imgs=True)
    dataloader.get_batch(32, "train")

