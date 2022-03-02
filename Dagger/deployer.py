from Data_collection.oculus_reader.reader import OculusReader
from env import YumiFullMoveEnv
import time
import numpy as np
import torch

from DataLoaders.dagger_loader import DaggerDataLoader
from trainer import DaggerTrainer

'''
DEPLOYMENT INSTRUCTION:
    1) LOAD BEST il MODEL
    2) PRESS RIGHT TRIGGER TO START ROLLOUT
    3) KEEP RIGHT TRIGGER PRESSED TO OVERRIDE POLICY AND RECORD NEW DATA
    4) PRESS RIGHT A (AT ANY TIME) TO FINISH ROLLOUT AND TRAIN ON AGGREGATED DATASET (THIS WILL GO BACK TO POINT 2)
'''


def main():

    transform_type = 2  # 0, 1, 2
    apply_normalization = True
    dim_a = 4
    model = 1  # 0: alexnet, 1: resnet18, 2: custom CNN
    device = 'cuda:0'

    dataset = DaggerDataLoader('../files/data/pick/dataset_IL', transform_type, apply_normalization, load_imgs=True)
    policy = DaggerTrainer(dataset, device, model, dim_a, transform_type, apply_normalization)
    policy.load_model('../files/models/pick/IL/best_il_model_prec=0.0116.mdl')

    oculus_reader = OculusReader()

    relative_pos_t = np.zeros(3)
    relative_pos_t1 = np.zeros(3)

    MIN_DISCRETIZATION = 0.01
    MAX_ABS_VEL = 0.1
    SCALE_VEL = 0.5

    n_trj = 0

    counter = 0
    while True:

        raw_data = oculus_reader.get_transformations_and_buttons()
        print("waiting for oculus aknowledgment ...")
        while not raw_data[0]:
            raw_data = oculus_reader.get_transformations_and_buttons()
            time.sleep(0.1)
        print("waiting for trigger ...")
        while not raw_data[1]['RTr']:
            raw_data = oculus_reader.get_transformations_and_buttons()
            relative_pos_t = raw_data[0]['r'][:3, 3]
            time_stamp_t = time.time()
            time.sleep(0.1)

        env = YumiFullMoveEnv(n_trj, dagger=True)

        # open gripper
        img = torch.zeros(1, 3, dataset.image_size_h, dataset.image_size_w).to(device)
        action = np.zeros(7)
        for _ in range(20):
            action[-1] = 1
            _, state = env.step(action)
        for _ in range(20):
            action[-1] = 0
            _, state = env.step(action)
        img_raw = env.get_frame()
        img[0] = dataset.process_img(img_raw)

        trajectory_complete = False
        while not trajectory_complete:

            img_raw = env.get_frame()
            img[0] = dataset.process_img(img_raw)

            raw_data = oculus_reader.get_transformations_and_buttons()
            relative_pos_t1 += raw_data[0]['r'][:3, 3]
            time_stamp_t1 = time.time()
            counter += 1

            if not raw_data[0] or raw_data[1]['A']:
                print("controller disconnected")
                trajectory_complete = True
            else:

                f_a = policy.deploy(img.to(device))
                f_a = f_a.cpu().detach().numpy()[0]
                action[:3] = f_a[:3]
                action[-1] = f_a[-1]

                if raw_data[1]['RTr']:

                    if raw_data[1]['RG']:
                        gripper_closed = True
                    else:
                        gripper_closed = False

                    relative_pos_t1 /= counter

                    pos_diff = (relative_pos_t1 - relative_pos_t) / (time_stamp_t1 - time_stamp_t)
                    pos_diff *= SCALE_VEL
                    pos_diff_threshold = (np.abs(pos_diff) > MIN_DISCRETIZATION) * pos_diff
                    relative_vel = np.clip(pos_diff_threshold, -MAX_ABS_VEL, MAX_ABS_VEL)

                    action[0] = -relative_vel[2]
                    action[1] = -relative_vel[0]
                    action[2] = relative_vel[1]
                    action[-1] = gripper_closed * 1.

                _, state = env.step(action)

                relative_pos_t = relative_pos_t1.copy()
                relative_pos_t1 = np.zeros(3)
                time_stamp_t = time_stamp_t1
                counter = 0

        env.release()

        new_folder = '../files/data/pick/dataset_IL_dagger/' + str(n_trj)
        dataset.aggregate(new_folder)

        policy.train(10)


if __name__ == '__main__':
    main()
