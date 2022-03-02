from behavior_cloning import BCTrainer
from equivariant_encoder import E_encoder
from gmm import Estimator
from DataLoaders.bc_data_loader import BCDataLoader
from DataLoaders.E_data_loader import EDataLoader
import argparse
import torch
import time


def train_E(dataloader_e, args, transform_type, task):

    E = E_encoder(transform_type, args.device, task=task)
    E.fit_E(dataloader_e, save_model=True)


def generate_trj(dataloader_bc, args, transform_type, task):

    E = E_encoder(transform_type, args.device, task=task)
    E.generate_equivariant_trajectories(dataloader_bc)


def train_gmm(dataloader_bc, args, transform_type, task):

    gmm = Estimator(transform_type, args.device, task=task)
    gmm.fit_gmm(dataloader_bc, save_model=True)


def train_bc(dataloader, args, transform_type, apply_normalization, task):
    dim_a = 4
    model = 1  # 0: alexnet, 1: resnet18, 2: custom CNN
    save_dir = '../files/models/push/IL' if task == "push" else '../files/models/pick/IL'
    bc_policy_config = {'device': args.device,
                        'num_epoch': args.num_epoch,
                        'snapshot': 20,
                        'batch_size': 50,
                        'save_dir': save_dir}

    bc_policy = BCTrainer(dataloader, bc_policy_config, model, dim_a, transform_type, apply_normalization, task)
    bc_policy.dataloader = dataloader
    bc_policy.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trainer')
    parser.add_argument('--device', default='cuda:0', type=str, help='the device for training, cpu or cuda')
    parser.add_argument('--num-epoch', default=5000, type=int, help='the number of epochs to train the model')
    args = parser.parse_args()

    task = "push"

    transform_type = 2  # 0, 1, 2
    apply_normalization = True

    dataloader_bc = BCDataLoader('../files/data/pick/dataset_IL', transform_type, apply_normalization, load_imgs=True)
    dataloader_e = EDataLoader('../files/data/equivariance/dataset_equivariance', transform_type, apply_normalization, load_imgs=False)

    # train_bc(dataloader_bc, args, transform_type, apply_normalization, task)
    # train_E(dataloader_e, args, transform_type, task)
    # generate_trj(dataloader_bc, args, transform_type, task)
    # train_gmm(dataloader_bc, args, transform_type, task)
