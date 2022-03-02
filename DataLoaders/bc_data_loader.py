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


def create_dataset(root, folder):
    dataset_path_raw = os.path.join(root, folder+'_raw')
    dataset_path = os.path.join(root, folder)
    if not os.path.exists(dataset_path):
        list_folders = os.listdir(dataset_path_raw)
        os.mkdir(dataset_path)
        actions = None
        states = None
        file_idx = 0
        for f in list_folders:
            action = np.load(os.path.join(dataset_path_raw, f) + '/actions.npy')
            state = np.load(os.path.join(dataset_path_raw, f) + '/states.npy')
            n_data = action.shape[0]

            dim_action = action.shape[1]
            dim_state = state.shape[1]
            if actions is None:
                actions = action
                states = state
            else:
                actions = np.concatenate((actions, action), 0)
                states = np.concatenate((states, state), 0)
            for i in range(n_data):
                src_fname = os.path.join(dataset_path_raw, f) + '/img{:04d}.png'.format(i)
                dst_fname = dataset_path + '/img{:04d}.png'.format(file_idx)
                file_idx += 1
                copyfile(src_fname, dst_fname)
        np.save(os.path.join(dataset_path, 'actions.npy'), actions)
        np.save(os.path.join(dataset_path, 'states.npy'), states)

        init_states = np.zeros((actions.shape[0]))
        st = 10.
        for t in range(actions.shape[0]):
            if np.abs(st - states[t, 0]) > 0.5:
                init_states[t] = 1
            st = states[t, 0]

        if np.sum(init_states) == len(list_folders):
            np.save(os.path.join(dataset_path, 'init_states.npy'), init_states)
        else:
            print("wrong init states")
            exit()


class BCDataLoader:
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
        self.actions = torch.from_numpy(np.load(os.path.join(self.dir, 'actions.npy'))).float()
        self.actions = torch.cat([self.actions[:, :3], self.actions[:, -1:]], -1)
        self.states = torch.from_numpy(np.load(os.path.join(self.dir, 'states.npy'))).float()
        self.init_states = np.load(os.path.join(self.dir, 'init_states.npy'))

        idx_init = np.nonzero(self.init_states)[0]
        self.groups = np.zeros(self.init_states.shape[0])
        self.img0 = torch.zeros(len(idx_init), 3, self.image_size_h, self.image_size_w)
        for j, i in enumerate(idx_init):
            self.img0[j] = self.process_img(PIL.Image.open(os.path.join(self.dir, 'img{:04d}.png'.format(i))))
        idx_init = np.append(idx_init, self.init_states.shape[0])
        for i in range(1, len(idx_init)):
            self.groups[idx_init[i-1]:idx_init[i]] = i-1

        self.load_imgs = load_imgs
        if self.load_imgs:
            self.imgs = torch.zeros(self.states.shape[0], 3, self.image_size_h, self.image_size_w)
            for i in tqdm(range(self.states.shape[0])):
                self.imgs[i] = self.process_img(PIL.Image.open(os.path.join(self.dir, 'img{:04d}.png'.format(i))))
            print("images successfully loaded in RAM")

        self.len = self.actions.shape[0]
        self.len_train = np.nonzero(self.init_states)[0][int(np.nonzero(self.init_states)[0].shape[0] * 0.8)]
        self.train_idx = np.arange(self.len_train)
        np.random.shuffle(self.train_idx)
        self.len_test = self.len - self.len_train
        self.test_idx = np.arange(self.len_train, self.len_train+self.len_test)
        np.random.shuffle(self.test_idx)

        self.train_ctr = 0
        self.test_ctr = 0

        self.n_trj = int(np.sum(self.init_states))
        self.trj = np.arange(self.n_trj)
        np.random.shuffle(self.trj)

        self.trj_ctr = 0

    def get_trajectory(self):
        if self.trj_ctr == self.n_trj:
            self.trj_ctr = 0
            np.random.shuffle(self.trj)
        idx0 = np.nonzero(self.init_states)[0][self.trj[self.trj_ctr]]
        if self.trj[self.trj_ctr] == self.n_trj - 1:
            idx1 = self.len
        else:
            idx1 = np.nonzero(self.init_states)[0][self.trj[self.trj_ctr]+1]
        idx = range(idx0, idx1)
        self.trj_ctr += 1

        idx = np.array(idx)
        imgs = self.get_img(idx)
        actions = self.actions[idx]
        states = self.states[idx]
        return imgs, actions, states

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

    def get_batch_gmm(self, batch_size, source):
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
        imgs0 = self.img0[self.groups[idx]]
        actions = self.actions[idx]

        return imgs, imgs0, actions[:, -1]

    def get_img(self, idx):

        if self.load_imgs:
            return self.imgs[idx]
        else:
            imgs = torch.zeros(idx.shape[0], 3, self.image_size_h, self.image_size_w)
            for i in range(idx.shape[0]):
                imgs[i] = self.process_img(PIL.Image.open(os.path.join(self.dir, 'img{:04d}.png'.format(idx[i]))))
            return imgs

    def get_normalized_params(self):
        means = []
        stds = []
        batch = 50
        n_batch = int(self.len_train/batch)
        for i in range(n_batch):
            img, _, _ = self.get_batch(batch, 'train')
            means.append([torch.mean(img[:][i]) for i in range(3)])
            stds.append([torch.std(img[:][i]) for i in range(3)])
        mean = torch.mean(torch.tensor(means), 0)
        std = torch.mean(torch.tensor(stds), 0)
        print(mean)
        print(std)

    def process_img(self, img):
        if self.apply_normalization:
            return self.normalize(self.transform(img))
        return self.transform(img)


if __name__ == '__main__':
    create_dataset('../files/data/pick/', 'dataset_IL')
    data_loader = BCDataLoader('../files/data/pick/dataset_IL', transform_type=2, apply_normalization=True, load_imgs=False)
    data_loader.get_trajectory()
    data_loader.get_normalized_params()

