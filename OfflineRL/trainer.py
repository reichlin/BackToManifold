import d3rlpy
import torch
import numpy as np
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from RL_data_loader import RLDataLoader
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(algorithm, params, device_flag, device):
    dataloader = RLDataLoader('../files/data/pick/dataset_IL', transform_type=2, apply_normalization=False, load_imgs=True)

    actor_model = d3rlpy.models.encoders.PixelEncoderFactory(filters=[(64, 3, 2),
                                                                      (64, 3, 2),
                                                                      (64, 3, 2),
                                                                      (8, 3, 2)],
                                                             feature_size=64,
                                                             activation='relu',
                                                             use_batch_norm=False,
                                                             dropout_rate=None)
    critic_model = d3rlpy.models.encoders.PixelEncoderFactory(filters=[(64, 3, 2),
                                                                       (64, 3, 2),
                                                                       (64, 3, 2),
                                                                       (8, 3, 2)],
                                                              feature_size=64,
                                                              activation='relu',
                                                              use_batch_norm=False,
                                                              dropout_rate=None)

    if algorithm == "cql":
        agent = d3rlpy.algos.CQL(actor_encoder_factory=actor_model,
                                 critic_encoder_factory=critic_model,
                                 batch_size=32,
                                 n_action_samples=4,
                                 use_gpu=device_flag)
        run_name = 'cql_seed='+str(params['seed'])
    elif algorithm == "plas":
        agent = d3rlpy.algos.PLAS(actor_encoder_factory=actor_model,
                                  critic_encoder_factory=critic_model,
                                  batch_size=32,
                                  use_gpu=device_flag)
        run_name = 'plas_seed=' + str(params['seed'])
    else:
        print("ERROR - no valid algorithm specified")
        exit()

    log_name = "./logs/"+run_name
    writer = SummaryWriter(log_name)
    min_precision = 1000.

    for epoch in range(10000):
        agent.fit(dataloader.dataset,
                  n_epochs=20,
                  experiment_name=None,#run_name,
                  logdir=run_name,
                  tensorboard_dir=None,#'./logs',
                  save_interval=101,
                  verbose=False,
                  # scorers={
                  #     'td_error': td_error_scorer,
                  #     'value_scale': average_value_estimation_scorer}
                  )
        agent.save_model(run_name+'.pt')

        # TEST MODEL
        mse = 0
        precision = 0
        n_batch = int(dataloader.len_train / 32)
        for _ in range(n_batch):
            imgs, actions = dataloader.get_batch(32, 'train')

            a_hat = agent.predict(imgs)

            mse += np.mean(np.sum((actions - a_hat) ** 2, -1))
            precision += np.mean(np.abs(actions[:, :3] - a_hat[:, :3]))

        Q = agent.predict_value(imgs, actions)[0]
        writer.add_scalar("train/Q", Q, epoch)

        mse /= n_batch
        precision /= n_batch

        writer.add_scalar("train/mse", mse, epoch)
        writer.add_scalar("train/precision", precision, epoch)


        mse = 0
        precision = 0
        n_batch = int(dataloader.len_test/32)
        for _ in range(n_batch):
            imgs, actions = dataloader.get_batch(32, 'test')

            a_hat = agent.predict(imgs)

            mse += np.mean(np.sum((actions - a_hat)**2, -1))
            precision += np.mean(np.abs(actions[:,:3] - a_hat[:,:3]))

        mse /= n_batch
        precision /= n_batch

        writer.add_scalar("test/mse", mse, epoch)
        writer.add_scalar("test/precision", precision, epoch)

        if min_precision > precision:
            min_precision = precision
            agent.save_model(run_name + '.pt')





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default=0, type=int, help='either cql of plas')
    parser.add_argument('--device', default=1, type=int, help='Use GPU')
    parser.add_argument('--seed', default=0, type=int, help='seed')

    args = parser.parse_args()
    params = args.__dict__
    if params['device'] == 1:
        device_flag = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device_flag = False
        device = torch.device('cpu')

    if params['algorithm'] == 0:
        algorithm = "cql"
    else:
        algorithm = "plas"

    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(algorithm, params, device_flag, device)



