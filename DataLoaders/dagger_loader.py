import torch
import torchvision.transforms as transforms
import PIL
import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm


class DaggerDataLoader:
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

        self.actions = torch.from_numpy(np.load(os.path.join(self.dir, 'actions.npy'))).float()
        self.actions = torch.cat([self.actions[:, :3], self.actions[:, -1:]], -1)
        self.states = torch.from_numpy(np.load(os.path.join(self.dir, 'states.npy'))).float()
        self.folders = []

        if self.load_imgs:
            self.imgs = torch.zeros(self.states.shape[0], 3, self.image_size_h, self.image_size_w)
            for i in tqdm(range(self.states.shape[0])):
                self.imgs[i] = self.process_img(PIL.Image.open(os.path.join(self.dir, 'img{:04d}.png'.format(i))))
            print("images successfully loaded in RAM")
        else:
            self.folders = [self.dir] * self.actions.shape[0]

        self.len = self.actions.shape[0]
        idx = np.arange(self.len)
        np.random.shuffle(idx)
        self.len_train = int(self.len * 0.8)
        self.train_idx = idx[:self.len_train]
        self.len_test = self.len - self.len_train
        self.test_idx = idx[self.len_train:]

        self.train_ctr = 0
        self.test_ctr = 0


    def aggregate(self, new_folder):

        actions = torch.from_numpy(np.load(os.path.join(new_folder, 'actions.npy'))).float()
        actions = torch.cat([actions[:, :3], actions[:, -1:]], -1)
        states = torch.from_numpy(np.load(os.path.join(new_folder, 'states.npy'))).float()

        self.actions = torch.cat([self.actions, actions], 0)
        self.states = torch.cat([self.states, states], 0)

        if self.load_imgs:
            imgs = torch.zeros(states.shape[0], 3, self.image_size_h, self.image_size_w)
            for i in tqdm(range(states.shape[0])):
                imgs[i] = self.process_img(PIL.Image.open(os.path.join(new_folder, 'img{:04d}.png'.format(i))))

            self.imgs = torch.cat([self.imgs, imgs], 0)
        else:
            folders = [new_folder] * actions.shape[0]
            self.folders.extend(folders)

        self.len = self.actions.shape[0]
        idx = np.arange(self.len)
        np.random.shuffle(self.test_idx)
        self.len_train = int(self.len * 0.8)
        self.train_idx = idx[:self.len_train]
        self.len_test = self.len - self.len_train
        self.test_idx = idx[self.len_train:]

        self.train_ctr = 0
        self.test_ctr = 0

    def get_batch(self, batch_size, source):
        if source == 'train':
            idx = self.train_idx[self.train_ctr: min(self.train_ctr + batch_size, self.len_train)]
            if (self.train_ctr + batch_size) > self.len_train:
                np.random.shuffle(self.train_idx)
                idx = np.append(idx, self.train_idx[0:(self.train_ctr + batch_size - self.len_train)])
                self.train_ctr = (self.train_ctr + batch_size - self.len_train)
            else:
                self.train_ctr += batch_size
        elif source == 'test':
            idx = self.test_idx[self.test_ctr: min(self.test_ctr + batch_size, self.len_test)]
            if (self.test_ctr + batch_size) > self.len_test:
                np.random.shuffle(self.test_idx)
                idx = np.append(idx, self.test_idx[0:(self.test_ctr + batch_size - self.len_test)])
                self.test_ctr = (self.test_ctr + batch_size - self.len_test)
            else:
                self.test_ctr += batch_size

        imgs = self.get_img(idx)
        actions = self.actions[idx]
        states = self.states[idx]
        return imgs, actions, states

    def get_img(self, idx):

        if self.load_imgs:
            return self.imgs[idx]
        else:
            imgs = torch.zeros(idx.shape[0], 3, self.image_size_h, self.image_size_w)
            for i in range(idx.shape[0]):
                imgs[i] = self.process_img(PIL.Image.open(os.path.join(self.folders[idx[i]], 'img{:04d}.png'.format(idx[i]))))
            return imgs

    def process_img(self, img):
        if self.apply_normalization:
            return self.normalize(self.transform(img))
        return self.transform(img)


