from d3rlpy.dataset import MDPDataset
import torch
import torchvision.transforms as transforms
import PIL
import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm


class RLDataLoader:
    def __init__(self, dataset_dir, transform_type=0, apply_normalization=False, load_imgs=False):

        self.apply_normalization = apply_normalization
        self.load_imgs = load_imgs

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
                                                 transforms.Resize((self.image_size_h, self.image_size_w)),]) #transforms.ToTensor()])

        if self.load_imgs:

            actions = np.load(os.path.join(dataset_dir, 'actions.npy'))
            self.actions = np.concatenate((actions[:, :3], actions[:, -1:]), -1)
            states = np.load(os.path.join(dataset_dir, 'states.npy'))
            init_states = np.load(os.path.join(dataset_dir, 'init_states.npy'))
            terminal_states = np.zeros(init_states.shape)
            terminal_states[-1] = 1
            n_datapoints = init_states.shape[0]
            for i in range(n_datapoints-1, 0, -1):
                if init_states[i] == 1:
                    terminal_states[i-1] = 1

            reward = np.zeros(terminal_states.shape)
            for i in range(n_datapoints - 1, 0, -1):
                if terminal_states[i] == 1:
                    reward[i] = 1
                else:
                    reward[i] = reward[i + 1] * 0.9

            self.imgs = np.zeros((states.shape[0], 3, self.image_size_h, self.image_size_w), dtype=np.uint8)
            for i in tqdm(range(states.shape[0])):
                self.imgs[i] = np.transpose(np.array(self.process_img(PIL.Image.open(os.path.join(dataset_dir, 'img{:04d}.png'.format(i))))).astype(np.uint8), (2, 0, 1))
            print("images successfully loaded in RAM")

            idxs = np.arange(states.shape[0])
            np.random.shuffle(idxs)
            self.train_idx = idxs[:int(states.shape[0]*0.8)]
            self.train_ctr = 0
            self.len_train = len(self.train_idx)

            self.test_idx = idxs[int(states.shape[0] * 0.8):]
            self.test_ctr = 0
            self.len_test = len(self.test_idx)

            self.dataset = MDPDataset(self.imgs[self.train_idx], self.actions[self.train_idx], reward[self.train_idx], terminal_states[self.train_idx].astype(bool))

        else:

            print("ERROR - load_img = False not yet implemented")
            exit()

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
        return imgs, actions

    def get_img(self, idx):

        if self.load_imgs:
            return self.imgs[idx]
        else:
            imgs = torch.zeros(idx.shape[0], 3, self.image_size_h, self.image_size_w)
            for i in range(idx.shape[0]):
                imgs[i] = self.process_img(PIL.Image.open(os.path.join(self.dir, 'img{:04d}.png'.format(idx[i]))))
            return imgs

    def process_img(self, img):
        if self.apply_normalization:
            return self.normalize(self.transform(img))
        return self.transform(img)


if __name__ == '__main__':
    data_loader = RLDataLoader('../files/data/pick/dataset_IL', transform_type=2, apply_normalization=False, load_imgs=True)