from env import YumiFullMoveEnv
from pynput import keyboard
import numpy as np
from behavior_cloning import BCTrainer
from DataLoaders.bc_data_loader import BCDataLoader
import argparse
import torch
from gmm import Estimator
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def on_press(key):
    try:
        global key_pressed
        key_pressed = key.char
    except AttributeError:
        pass


MAX_ABS_VEL = 0.1
model = 1
dim_a = 4
transform_type = 2
apply_normalization = True

robust = False
perturbation = True
use_adam = False

task = "pick"  # either "pick" or "push"

parser = argparse.ArgumentParser(description='trainer')
parser.add_argument('--device', default='cpu', type=str, help='the device for training, cpu or cuda')
args = parser.parse_args()

dataloader = BCDataLoader('../files/data/pick/dataset_IL', transform_type, apply_normalization, load_imgs=False)

bc_policy_config = {'device': args.device,
                    'num_epoch': 1,
                    'snapshot': 20,
                    'batch_size': 50,
                    'save_dir': '../files/models/pick/IL'}
bc_policy = BCTrainer(dataloader, bc_policy_config, model, dim_a, transform_type, apply_normalization, task)
if task == "pick":
    bc_policy.load_model('../files/models/pick/IL/best_il_model_prec=0.0116.mdl')
elif task == "push":
    bc_policy.load_model('../files/models/push/IL/best_il_model_prec=0.0072.mdl')

density_estimator = Estimator(transform_type, args.device, task)



imgs, actions, states = dataloader.get_trajectory()
z = density_estimator.E(imgs)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(z_vals[:, 0], z_vals[:, 1], z_vals[:, 2], s=[v*10 for v in p_vals])
# ax.plot(z_vals[:, 0], z_vals[:, 1], z_vals[:, 2])

plt.show()






img = torch.zeros(1, 3, bc_policy.dataloader.image_size_h, bc_policy.dataloader.image_size_w).to(args.device)
listener = keyboard.Listener(on_press=on_press)
listener.start()
env = YumiFullMoveEnv()
key_pressed = ''


print("ready to deploy")
time.sleep(3)

# delta_p = 0.1
eta_g = 0.05

alpha = 0.02
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
moment_m = np.zeros(3)
moment_v = np.zeros(3)

z_vals = None
rho_vals = []
p_vals = []

# open gripper
action = np.zeros(7)
for _ in range(20):
    action[-1] = 1
    _, state = env.step(action)
if task == "pick":
    for _ in range(20):
        action[-1] = 0
        _, state = env.step(action)

# fig = plt.figure()
imgs, actions, states = dataloader.get_trajectory()
calibrated = False
while not calibrated:
    img_raw = env.get_frame()
    img[0] = bc_policy.dataloader.process_img(img_raw)

    log_p, zt = density_estimator.get_probability_image(img, img, torch.zeros(1, 1).to(args.device))
    zt = zt[0].detach().cpu().numpy()
    print(zt)

    # diff_img = torch.abs(img[0] - imgs[0])
    # plt.imshow(torch.transpose(torch.transpose(diff_img, 1, 0), 2, 1).detach())
    # plt.show(block=False)
    # plt.pause(0.05)

    if np.sum(zt**2) < 0.005:
        calibrated = True

for _ in range(5):
    _ = env.get_frame()

img_raw = env.get_frame()
img[0] = bc_policy.dataloader.process_img(img_raw)
o0 = img * 1.
grasp = torch.zeros(1, 1).to(args.device) if task == "pick" else torch.ones(1, 1).to(args.device)

mu_t, sigma_t, w_t, _ = density_estimator.GMM(o0.repeat(grasp.shape[0], 1, 1, 1), grasp)
mu = mu_t.detach().cpu().numpy()
sigma = torch.diag_embed(sigma_t).detach().cpu().numpy()
w = w_t.detach().cpu().numpy()




# from torch.distributions.multivariate_normal import MultivariateNormal
# from torch.distributions.categorical import Categorical
# from torch.distributions.mixture_same_family import MixtureSameFamily
# from tqdm import tqdm
#
# zx = np.linspace(-0.1, 1.1, 120)
# zy = np.linspace(-0.1, 1.6, 170)
# zz = np.linspace(-0.3, 0.3, 10)
# density = np.zeros((120, 170, 10))
#
# comp = MultivariateNormal(mu_t, torch.diag_embed(sigma_t))
# mix = Categorical(w_t)
# gmm = MixtureSameFamily(mix, comp)
#
# with torch.no_grad():
#     for i in tqdm(range(120)):
#         for j in range(170):
#             for k in range(10):
#                 z = torch.zeros(1, 3).to(args.device)
#                 z[0, 0] = zx[i]
#                 z[0, 1] = zy[j]
#                 z[0, 2] = zz[k]
#                 density[i, j, k] = torch.exp(gmm.log_prob(z)).cpu().item()
#
# print()
#
# plt.imshow(np.mean(density, -1))










theta_perturbation = np.random.rand() * np.pi/2.

Fx = np.cos(theta_perturbation)
Fy = np.sin(theta_perturbation)
perturbation_t = 35

# bypass = True

for t in range(1000):

    if key_pressed == 'z':
        break

    # if key_pressed == 'a':
    #     bypass = False

    img_raw = env.get_frame()
    img[0] = bc_policy.dataloader.process_img(img_raw)

    f_a = bc_policy.deploy(img.to(args.device))
    f_a = f_a.cpu().detach().numpy()[0]

    log_p, zt, g_a = density_estimator.get_recovery_action(o0, img, grasp)
    norm_g_a = np.clip(np.linalg.norm(g_a), a_min=1e-20, a_max=None)
    g_a = eta_g * (g_a / norm_g_a)

    off = 0.5 #1.0 #1.0 #2.0
    tau = 0.5
    p = torch.sigmoid((log_p - off) / tau).detach().cpu().item()
    print(p)

    if z_vals is None:
        z_vals = zt.detach().cpu().numpy()
        p_vals.append(p)
        rho_vals.append(log_p.detach().cpu().item())
    else:
        z_vals = np.concatenate((z_vals, zt.detach().cpu().numpy()), axis=0)
        p_vals.append(p)
        rho_vals.append(log_p.detach().cpu().item())

    if robust:

        if use_adam:
            moment_m = beta1 * moment_m + (1-beta1) * g_a[:3]
            moment_v = beta2 * moment_v + (1-beta2) * (g_a[:3] ** 2)
            moment_m_hat = moment_m / (1 - beta1)
            moment_v_hat = moment_v / (1 - beta2)
            momentum = alpha * moment_m_hat / (np.sqrt(moment_v_hat) + epsilon)
        else:
            momentum = g_a[:3]

        # if bypass:
        #     p = 1.

        action[:3] = p * f_a[:3] + (1-p) * momentum
        action[-1] = f_a[-1]
        # print(p, momentum)
    else:
        action[:3] = f_a[:3]
        action[-1] = f_a[-1]

    action[:3] = np.clip(action[:3], -MAX_ABS_VEL, MAX_ABS_VEL) / 2.

    if perturbation:
        if 0 < t <= perturbation_t:
            print("perturbation")
            action[0] = Fx * 0.03
            action[1] = Fy * 0.03
            action[2] = 0.0
            action[-1] = 0.
        elif perturbation_t < t < perturbation_t+10:
            print("stop")
            action[0] = 0.0
            action[1] = 0.0
            action[2] = 0.0
            action[-1] = 0.

    if task == "push":
        action[-1] = 1.
    grasp[0, 0] = (action[-1] > 0.5) * 1

    _, state = env.step(action, deploy=True)

env.release()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(z_vals[:, 0], z_vals[:, 1], z_vals[:, 2], s=[v*10 for v in p_vals])
ax.plot(z_vals[:, 0], z_vals[:, 1], z_vals[:, 2])

for n in range(density_estimator.N):
    if w[0, n] > 0.01:
        center = mu[0, n]
        A = sigma[0, n]
        U, s, rotation = np.linalg.svd(A)
        radii = 1.0 / np.sqrt(s) * 0.001
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
        ax.plot_surface(x, y, z, rstride=3, cstride=3, linewidth=0.1, alpha=0.2, shade=True)
plt.show()

print()