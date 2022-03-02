import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, dim_input, dim_output, transform_type):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        if transform_type == 0:
            self.dim_f = 256 * 7 * 7
        elif transform_type == 1:
            self.dim_f = 256 * 5 * 7
        elif transform_type == 2:
            self.dim_f = 256 * 6 * 6

        if self.dim_input > 3:
            self.pre_conv = nn.Conv2d(dim_input, 3, kernel_size=3, stride=1, padding=1)

        alexnet = models.alexnet(pretrained=True)
        self.base_model = alexnet.features
        self.fc1 = nn.Linear(self.dim_f, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.dim_output)

    def forward(self, x):

        if self.dim_input > 3:
            x = F.relu(self.pre_conv(x))

        x = self.base_model(x)
        x = x.view(-1, self.dim_f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_loss(self, actions, actions_predicted):

        loss_v = torch.mean(torch.sum((actions[:, :3] - actions_predicted[:, :3]) ** 2, -1))
        loss_g = torch.mean(torch.sum((actions[:, -1] - actions_predicted[:, -1]) ** 2, -1))
        loss = 100 * loss_v + loss_g

        return loss


class ResNet18(nn.Module):
    def __init__(self, dim_input, dim_output, transform_type):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        if transform_type == 0:
            self.dim_f = 512 * 8 * 8
        elif transform_type == 1:
            self.dim_f = 512 * 6 * 8
        elif transform_type == 2:
            self.dim_f = 512 * 7 * 7

        if self.dim_input > 3:
            self.pre_conv = nn.Conv2d(dim_input, 3, kernel_size=3, stride=1, padding=1)

        resnet = models.resnet18(pretrained=True)
        self.base_model = resnet
        self.fc1 = nn.Linear(self.dim_f, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.dim_output)

    def forward(self, x):

        if self.dim_input > 3:
            x = F.relu(self.pre_conv(x))

        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = x.view(-1, self.dim_f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_loss(self, actions, actions_predicted):

        loss_v = torch.mean(torch.sum((actions[:, :3] - actions_predicted[:, :3])**2, -1))
        loss_g = torch.mean(torch.sum((actions[:, -1] - actions_predicted[:, -1])**2, -1))
        loss = 100*loss_v + loss_g

        return loss


class CNN(nn.Module):
    def __init__(self, dim_input, dim_output, transform_type):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        if transform_type == 0:
            self.dim_f = 64 * 15 * 15
        elif transform_type == 1:
            self.dim_f = 64 * 11 * 15
        elif transform_type == 2:
            self.dim_f = 8 * 13 * 13

        self.conv1 = nn.Conv2d(dim_input, 64, kernel_size=3, stride=2, padding=0)
        # self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        # self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        # self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 8, kernel_size=3, stride=2, padding=0)
        # self.bn4 = torch.nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(self.dim_f, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.dim_output)

    def forward(self, x):

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.dim_f)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_loss(self, actions, actions_predicted):

        loss_v = torch.mean(torch.sum((actions[:, :3] - actions_predicted[:, :3]) ** 2, -1))
        loss_g = torch.mean(torch.sum((actions[:, -1] - actions_predicted[:, -1]) ** 2, -1))
        loss = 100 * loss_v + loss_g

        return loss


# class RNN(nn.Module):
#
#     def __init__(self, dim_input, dim_output, dim_h, transform_type):
#         super().__init__()
#         self.dim_input = dim_input
#         self.dim_h = dim_h
#         self.dim_output = dim_output
#         if transform_type == 0:
#             self.dim_f = 512 * 8 * 8
#         elif transform_type == 1:
#             self.dim_f = 512 * 6 * 8
#         elif transform_type == 2:
#             self.dim_f = 512 * 7 * 7
#
#         if self.dim_input > 3:
#             self.pre_conv = nn.Conv2d(dim_input, 3, kernel_size=3, stride=1, padding=1)
#
#         resnet = models.resnet18(pretrained=True)
#         self.base_model = resnet  # .features #alexnet.features
#         # self.conv = nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(self.dim_f, 32)
#
#         self.rnn_fc1 = nn.Linear(32 + dim_h, 64)
#         self.rnn_fc2 = nn.Linear(64, 64)
#         self.rnn_fc3 = nn.Linear(64, self.dim_output + dim_h)
#
#     def forward(self, x, h):
#
#         if self.dim_input > 3:
#             x = F.relu(self.pre_conv(x))
#
#         x = self.base_model.conv1(x)
#         x = self.base_model.bn1(x)
#         x = self.base_model.relu(x)
#         x = self.base_model.maxpool(x)
#
#         x = self.base_model.layer1(x)
#         x = self.base_model.layer2(x)
#         x = self.base_model.layer3(x)
#         x = self.base_model.layer4(x)
#
#         x = x.view(-1, self.dim_f)
#         x = F.relu(self.fc1(x))
#
#         x = torch.cat([x, h], -1)
#
#         x = F.relu(self.rnn_fc1(x))
#         x = F.relu(self.rnn_fc2(x))
#         x = self.rnn_fc3(x)
#
#         a = x[:, :self.dim_output]
#         h = x[:, self.dim_output:]
#
#         return a, h
#
#     def get_loss(self, actions, actions_predicted):
#
#         loss = F.mse_loss(actions, actions_predicted)
#
#         return loss









class Equivariant_map_images(nn.Module):

    def __init__(self, dim_input_o, dim_output, transform_type):
        super().__init__()

        if transform_type == 0:
            self.dim_f = 64 * 15 * 15
        elif transform_type == 1:
            self.dim_f = 64 * 11 * 15
        elif transform_type == 2:
            self.dim_f = 8 * 6 * 6

        self.conv1 = nn.Conv2d(dim_input_o, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 8, kernel_size=3, stride=2, padding=0)

        self.fc1 = nn.Linear(self.dim_f, 64)
        self.fc2 = nn.Linear(64, dim_output)

    def forward(self, o):

        z = F.relu(self.conv1(o))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))

        z = F.relu(self.conv5(z))

        z = z.view(-1, self.dim_f)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)

        return z


class Mixture_Density_Network(nn.Module):

    def __init__(self, N, d, transform_type):
        super().__init__()

        self.d = d
        self.N = N

        if transform_type == 0:
            self.dim_f = 64 * 15 * 15
        elif transform_type == 1:
            self.dim_f = 64 * 11 * 15
        elif transform_type == 2:
            self.dim_f = 8 * 6 * 6

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 8, kernel_size=3, stride=2, padding=0)

        self.fc1 = nn.Linear(self.dim_f, 16)
        self.fc2 = nn.Linear(16+1, 16)

        self.means = nn.Linear(16, self.d * N)
        self.stds = nn.Linear(16, self.d * N)
        self.w = nn.Linear(16, N)

        self.sm = nn.Softmax(dim=1)

        self.deconv1 = nn.ConvTranspose2d(8, 64, 3, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0)
        self.deconv5 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=0)

        # if transform_type == 0:
        #     self.dim_f = 512 * 8 * 8
        # elif transform_type == 1:
        #     self.dim_f = 512 * 6 * 8
        # elif transform_type == 2:
        #     self.dim_f = 512 * 7 * 7
        #
        # resnet = models.resnet18(pretrained=False)
        # self.base_model = resnet
        # self.fc1 = nn.Linear(self.dim_f, 16)
        # self.fc2 = nn.Linear(16 + 1, 16)
        #
        # self.means = nn.Linear(16, self.d*N)
        # self.stds = nn.Linear(16, self.d*N)
        #
        # self.w = nn.parameter.Parameter(torch.ones((1, N)) * 1./N, requires_grad=False)

    def forward(self, o, g):

        z = F.relu(self.conv1(o))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))

        h = F.relu(self.conv5(z))

        z = h.view(-1, self.dim_f)
        z = F.relu(self.fc1(z))
        z = torch.cat([z, g], -1)
        z = self.fc2(z)

        mu = self.means(z).view(-1, self.N, self.d)
        sigma = F.elu(self.stds(z).view(-1, self.N, self.d)) + 1 + 0.00001
        w = self.sm(self.w(z))

        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))

        o_hat = self.deconv5(h)

        return mu, sigma, w, o_hat

        # x = self.base_model.conv1(o)
        # x = self.base_model.bn1(x)
        # x = self.base_model.relu(x)
        # x = self.base_model.maxpool(x)
        #
        # x = self.base_model.layer1(x)
        # x = self.base_model.layer2(x)
        # x = self.base_model.layer3(x)
        # x = self.base_model.layer4(x)
        #
        # x = x.view(-1, self.dim_f)
        # x = F.relu(self.fc1(x))
        # x = torch.cat([x, g], -1)
        # x = F.relu(self.fc2(x))
        #
        # mu = self.means(x).view(-1, self.N, self.d)
        # sigma = torch.sigmoid(self.stds(x).view(-1, self.N, self.d)) * 100 + 0.00001
        #
        # elu + 1 + epsilon
        #
        # return mu, sigma, self.w.repeat(mu.shape[0], 1)

    def decode(self, o):

        z = F.relu(self.conv1(o))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))

        z = F.relu(self.conv5(z))

        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))

        o_hat = self.deconv5(z)

        return o_hat