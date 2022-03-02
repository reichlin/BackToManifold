import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
from models import AlexNet, ResNet18, CNN

# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt


class BCTrainer:
    def __init__(self, dataloader, config, model=0, dim_a=4, transform_type=0, apply_normalization=False, task="pick"):
        self.device = config['device']
        self.num_epoch = config['num_epoch']
        self.epoch_to_save = self.snapshot = config['snapshot']
        self.batch_size = config['batch_size']
        self.save_dir = config['save_dir']
        self.dim_imgs = 3
        self.model = model
        self.dim_a = dim_a

        if model == 0:  # alexnet
            self.policy = AlexNet(self.dim_imgs, dim_a, transform_type).to(self.device)
        elif model == 1:  # resnet18
            self.policy = ResNet18(self.dim_imgs, dim_a, transform_type).to(self.device)
        elif model == 2:  # custom CNN
            self.policy = CNN(self.dim_imgs, dim_a, transform_type).to(self.device)

        self.dataloader = dataloader
        self.lr = 1e-4 #1e-3
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.epoch = 0

        self.name_run = "1"
        log_name = "../logs/logs_il/push" if task == "push" else "../logs/logs_il/pick"

        self.writer = SummaryWriter(log_name+self.name_run)
        self.log_idx = 0

        self.best_eval_loss = None
        self.best_precision_loss = None

    def train(self):

        for self.epoch in tqdm(range(self.num_epoch)):
            self.policy.train()
            sum_loss = 0

            for _ in range(int(self.dataloader.len_train / self.batch_size)+1):
                self.optimizer.zero_grad()
                imgs, actions, states = self.dataloader.get_batch(self.batch_size, 'train')
                imgs, actions, states = imgs.to(self.device), actions.to(self.device), states.to(self.device)

                actions_predicted = self.policy(imgs)

                loss = self.policy.get_loss(actions, actions_predicted)

                loss.backward()
                self.optimizer.step()
                sum_loss += loss.detach().cpu().item()

            self.writer.add_scalar('Loss/train', sum_loss/(int(self.dataloader.len_train / self.batch_size)+1), self.log_idx)

            eval_loss, avg_diff = self.evaluate()
            precision = (avg_diff[0] + avg_diff[1] + avg_diff[2]) / 3.
            self.writer.add_scalar('Loss/test', eval_loss, self.log_idx)
            self.writer.add_scalar('Loss/test_precision', precision, self.log_idx)
            self.writer.add_scalar('Action_diff/x', avg_diff[0], self.log_idx)
            self.writer.add_scalar('Action_diff/y', avg_diff[1], self.log_idx)
            self.writer.add_scalar('Action_diff/z', avg_diff[2], self.log_idx)
            self.writer.add_scalar('Action_diff/gripper', avg_diff[-1], self.log_idx)
            self.log_idx += 1

            if self.best_eval_loss is None:
                self.best_eval_loss = eval_loss
            elif self.best_eval_loss >= eval_loss:
                self.save_model(eval_loss)
                self.best_eval_loss = eval_loss

            if self.best_precision_loss is None:
                self.best_precision_loss = precision
            elif self.best_precision_loss >= precision:
                self.save_model(precision, precision=True)
                self.best_precision_loss = precision

        self.writer.close()

    def evaluate(self):
        self.policy.eval()
        sum_loss = 0
        avg_diff = np.zeros(self.dim_a)

        for _ in range(int(self.dataloader.len_test / self.batch_size)+1):
            imgs, actions, states = self.dataloader.get_batch(self.batch_size, 'test')
            imgs, actions, states = imgs.to(self.device), actions.to(self.device), states.to(self.device)
            actions_predicted = self.policy(imgs)
            sum_loss += self.policy.get_loss(actions, actions_predicted).detach().cpu().numpy()
            avg_diff += torch.mean(torch.abs(actions - actions_predicted), 0).detach().cpu().numpy()

        return sum_loss/(int(self.dataloader.len_test / self.batch_size)+1), avg_diff/(int(self.dataloader.len_test / self.batch_size)+1)

    def deploy(self, img):
        self.policy.eval()
        actions = self.policy(img)
        return actions

    def save_model(self, metric, precision=False):

        save_dir = self.save_dir

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if not os.path.isdir(save_dir+"/"+self.name_run):
            os.makedirs(save_dir+"/"+self.name_run)

        if precision:
            fname = save_dir+"/"+self.name_run+'/epoch='+str(self.log_idx)+'_prec='+str(metric)+'.mdl'
        else:
            fname = save_dir+"/"+self.name_run+'/epoch='+str(self.log_idx)+'_loss='+str(metric)+'.mdl'

        torch.save(self.policy.state_dict(), fname)

    def load_model(self, filename):
        state_dict = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()



