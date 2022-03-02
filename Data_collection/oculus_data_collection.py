from oculus_reader.reader import OculusReader
from env import YumiFullMoveEnv
import time
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.linalg import logm

from DataLoaders.bc_data_loader import BCDataLoader
import torch
from Robust_IL.gmm import Estimator


def main():

    task_n = 59
    test_DE = False

    if test_DE:
        # ##############################################################################################################
        transform_type = 2
        apply_normalization = True

        device = "cuda:0"

        dataloader = BCDataLoader('../files/data/pick/dataset_IL', transform_type, apply_normalization)

        density_estimator = Estimator(transform_type, device, "pick")

        img = torch.zeros(1, 3, dataloader.image_size_h, dataloader.image_size_w).to("cuda:0")
        img0 = torch.zeros(1, 3, dataloader.image_size_h, dataloader.image_size_w).to("cuda:0")
        grasp = torch.zeros(1, 1).to("cuda:0")

        # ##############################################################################################################

    oculus_reader = OculusReader()
    env = YumiFullMoveEnv(task_n, False)

    relative_pos_t = np.zeros(3)
    relative_pos_t1 = np.zeros(3)

    relative_R_t = np.zeros((3, 3))
    relative_R_t1 = np.zeros((3, 3))

    MIN_DISCRETIZATION = 0.01 #0.0001
    MAX_ABS_VEL = 0.1
    SCALE_VEL = 0.5  # 1.0 #0.5

    MIN_ROT_DISCRETIZATION = 0.0001
    MAX_ABS_ROT = 0.2
    SCALE_ROT = 0.5 #0.4

    controller_ready = False
    while not controller_ready:
        raw_data = oculus_reader.get_transformations_and_buttons()
        if raw_data[0]:

            if test_DE:
                img_raw = env.get_frame()
                img0[0] = dataloader.process_img(img_raw)

            relative_pos_t = raw_data[0]['r'][:3, 3]
            # relative_rot_t = Rotation.from_matrix(raw_data[0]['r'][:3, :3]).as_rotvec() + np.pi
            relative_R_t = raw_data[0]['r'][:3, :3]
            time_stamp_t = time.time()
            if raw_data[1]['RTr']:
                controller_ready = True
        else:
            print("no sensible data from right controller")
            time.sleep(0.1)

    print("start connection with the robot")

    counter = 0
    angular_vel = np.zeros(3)
    gripper_closed = False
    state = None
    while True:
        raw_data = oculus_reader.get_transformations_and_buttons()
        relative_pos_t1 += raw_data[0]['r'][:3, 3]
        # relative_rot_t1 = Rotation.from_matrix(raw_data[0]['r'][:3, :3]).as_rotvec() + np.pi
        relative_R_t1 = raw_data[0]['r'][:3, :3]
        time_stamp_t1 = time.time()
        # print(relative_rot_t1)
        # time.sleep(1.)
        counter += 1

        if not raw_data[0] or raw_data[1]['A']:
            print("controller disconnected")
            break

        if env.ready:

            if raw_data[1]['RG']:
                gripper_closed = True
            else:
                gripper_closed = False

            if raw_data[1]['RTr']:
                action = np.zeros(7)
                action[-1] = gripper_closed * 1.
                env.step(action)
            else:

                relative_pos_t1 /= counter

                pos_diff = (relative_pos_t1 - relative_pos_t) / (time_stamp_t1 - time_stamp_t)
                pos_diff *= SCALE_VEL #3
                pos_diff_threshold = (np.abs(pos_diff) > MIN_DISCRETIZATION) * pos_diff
                relative_vel = np.clip(pos_diff_threshold, -MAX_ABS_VEL, MAX_ABS_VEL)

                # rot_diff = logm(np.matmul(relative_R_t1, relative_R_t.T)) / (time_stamp_t1 - time_stamp_t)
                # angular_vel[0] = 0.#rot_diff[0, 1]
                # angular_vel[1] = 0.#rot_diff[1, 2]
                # angular_vel[2] = 0.#rot_diff[0, 2] #  #-rot_diff[0, 1]
                #
                # angular_vel = (np.abs(angular_vel) > MIN_ROT_DISCRETIZATION) * angular_vel
                # angular_vel *= SCALE_ROT
                # angular_vel = np.clip(angular_vel, -MAX_ABS_ROT, MAX_ABS_ROT)

                # print("record action: " + str(relative_vel))
                action = np.zeros(7)
                action[0] = -relative_vel[2]
                action[1] = -relative_vel[0]
                action[2] = relative_vel[1]
                # action[3:6] = angular_vel
                action[-1] = 1. #gripper_closed * 1.

                if test_DE:
                    grasp[0] = gripper_closed * 1.
                    img_raw = env.get_frame()
                    img[0] = dataloader.process_img(img_raw)
                    p, _ = density_estimator.get_probability_image(img0, img, grasp)

                    print(p.detach().cpu().item())
                    # off = 0.005
                    # tau = 0.001
                    # print(torch.sigmoid((p - off) / tau).detach().cpu().item())
                    # print(p.detach().cpu().item())

                _, state = env.step(action)
                # print(action)

            relative_pos_t = relative_pos_t1.copy()
            relative_pos_t1 = np.zeros(3)
            relative_R_t = relative_R_t1.copy()
            time_stamp_t = time_stamp_t1
            counter = 0

    env.release()
    return


if __name__ == '__main__':

    main()
