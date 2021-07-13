import math

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch import autograd
import numpy as np
from torch.autograd import Variable
import utils
import cv2


class Learner(nn.Module):

    def __init__(self, net, alpha, *args):
        super(Learner, self).__init__()
        self.alpha = alpha

        self.net_theta = net()  # theta : prior / general
        self.net_phi = net()  # phi : task specific
        self.optimizer = optim.SGD(self.net_phi.parameters(), self.alpha)  # Learner(inner loop, for task specific phi)

        self.reg_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

    def forward(self, support_x, support_y, query_x, query_y, num_updates):
        # To get phi from current theta (fine tune)
        # copy theta to phi

        with torch.no_grad():
            for theta, phi in zip(self.net_theta.modules(), self.net_phi.modules()):
                if isinstance(phi, nn.Linear) or isinstance(phi, nn.Conv2d) or isinstance(phi, nn.BatchNorm2d):
                    phi.weight.data = theta.weight.clone()  # you must use .clone()
                    if phi.bias is not None:
                        phi.bias.data = theta.bias.clone()

        for i in range(num_updates):
            angles = self.net_phi(support_x, support_y)
            loss = self.reg_criterion(angles, support_y)
            loss_mae = self.mae_criterion(angles, support_y)
            if math.isnan(loss) or math.isnan(loss_mae):
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Calculating meta gradient
        # Calculate phi net's gradient to update theta by meta learner
        angles = self.net_phi(query_x, query_y)
        loss = self.reg_criterion(angles, query_y)
        loss_mae = self.mae_criterion(angles, query_y)

        # create_graph=True : Can recall backward after autograd.grad (for Hessian)
        gradient_phi = autograd.grad(loss, self.net_phi.parameters(), create_graph=True)  # create_graph : for second derivative

        return loss, loss_mae, gradient_phi

    def net_forward(self, support_x, support_y):
        # theta update (general)
        # To write the merged gradients in net_theta network from metalearner

        angles = self.net_theta(support_x, support_y)
        loss = self.reg_criterion(angles, support_y)

        return loss


class MetaLearner(nn.Module):
    # Received the loss of various tasks in net_pi network and found a general initialization parameter that combines everything.
    # Update theta by using phi and meta-test set for every episode

    def __init__(self, net, net_args, n_way, k_shot, meta_batch_size, alpha, beta, num_updates):
        super(MetaLearner, self).__init__()

        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_batch_size = meta_batch_size
        self.beta = beta
        self.num_updates = num_updates

        self.learner = Learner(net, alpha, *net_args)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)
        self.reg_criterion = nn.MSELoss()

    def meta_update(self, dummy_loss, sum_grads_phi):
        # Update theta_parameter by sum_gradients
        hooks = []
        for k, v in enumerate(self.learner.parameters()):
            def closure():
                key = k
                return lambda grad: sum_grads_phi[key]

            hooks.append(v.register_hook(closure()))
            # register_hook : If you manipulate the gradients, the optimizer will use these new custom gradients to update the parameters
            # If you want to save gradients
            # The purpose of this piece of code is to investigate how to use modified gradient to update parameters.

        self.optimizer.zero_grad()
        dummy_loss.backward() # dummy_loss : summed gradients_phi (for general theta network)
        self.optimizer.step()

        for h in hooks:
            h.remove()

    def forward(self, support_x, support_y, query_x, query_y):
        # Learned by Learner for every episode -> get the losses of parameter theta
        # Get loss and combine to update theta

        sum_grads_phi = None
        meta_batch_size = support_y.size(0)

        loss_values = []
        loss_values_mae = []
        for i in range(meta_batch_size):
            loss, mae, grad_phi = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            if math.isnan(loss) or math.isnan(mae):
                break
            loss_values.append(loss)
            loss_values_mae.append(mae)
            if sum_grads_phi is None:
                sum_grads_phi = grad_phi
            else:
                sum_grads_phi = [torch.add(i, j) for i, j in zip(sum_grads_phi, grad_phi) if i is not None]  # to get theta

            # print(f'MAML Training ---> Yaw Loss: {abs(loss_y)}, Pitch Loss: {abs(loss_p)}, Roll Loss: {abs(loss_r)}')

        # loss = self.learner.net_forward(support_x[0], support_y[0])

        return loss_values, loss_values_mae

    def pred(self, support_x, support_y, query_x, query_y):
        meta_batch_size = support_y.size(0)
        losses = []
        mae_losses = []
        for i in range(meta_batch_size):
            loss, loss_mae, grad_phi = self.learner(support_x[i], support_y[i], query_x[i], query_y[i],  self.num_updates)
            if math.isnan(loss) or math.isnan(loss_mae):
                break
            losses.append(loss)
            mae_losses.append(loss_mae)
            # img = utils.draw_axis(query_x[i][0], pred_y[i - 1], pred_p[i - 1], pred_r[i - 1])
            # plt.imshow(img[0])
            # plt.imshow(img[1])
            # plt.show()

        pred_loss = torch.mean(torch.stack(losses))
        pred_mae_loss = torch.mean(torch.stack(mae_losses))

        return pred_loss, pred_mae_loss