import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch import autograd
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import cv2


class Learner(nn.Module):

    def __init__(self, net, alpha, *args):
        super(Learner, self).__init__()
        self.alpha = alpha

        self.net_theta = net(*args)  # theta : prior / general
        self.net_phi = net(*args)  # phi : task specific
        self.optimizer = optim.SGD(self.net_phi.parameters(), self.alpha)  # Learner(inner loop, for task specific phi)
        self.criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        self.softmax = nn.Softmax()
        self.alpha = 0.001
        self.idx_tens = [idx for idx in range(66)]
        self.idx_tensor = Variable(torch.FloatTensor(self.idx_tens))


    def forward(self, support_x, support_y, support_cont, query_x, query_y, query_cont, num_updates):
        # To get phi from current theta (fine tune)
        # copy theta to phit


        with torch.no_grad():
            for theta, phi in zip(self.net_theta.modules(), self.net_phi.modules()):
                if isinstance(phi, nn.Linear) or isinstance(phi, nn.Conv2d) or isinstance(phi, nn.BatchNorm2d):
                    phi.weight.data = theta.weight.clone()  # you must use .clone()
                    if phi.bias is not None:
                        phi.bias.data = theta.bias.clone()
                        # clone():copy the data to another memory but it has no interfere with gradient back propagation (cf. deepcopy)

        # support_x: [5, 1, 28, 28]
        for i in range(num_updates):
            # loss, pred = self.net_phi(support_x, support_y)
            y, p, r = self.net_phi(support_x, support_y)

            loss_y, loss_p, loss_r, y_pred, p_pred, r_pred = self.get_loss(y,p,r, support_y, support_cont)
            loss_seq = [loss_y, loss_p, loss_r]

            grad_seq = [torch.tensor(1, dtype=float) for _ in range(len(loss_seq))]
            self.optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)

            self.optimizer.step()

            # print(f'Episode {i + 1}, Losses: Yaw: {loss_y.item()} , Pitch : {loss_p.item()}, Roll : {loss_r.item()}')

        # Calculating meta gradient
        # Calculate phi net's gradient to update theta by meta learner

        y, p, r = self.net_phi(query_x, query_y)
        loss_q_y, loss_q_p, loss_q_r, q_y_pred, q_p_pred, q_r_pred = self.get_loss(y, p, r, query_y, query_cont)
        loss_seq = [loss_q_y, loss_q_p, loss_q_r]

        # create_graph=True : Can recall backward after autograd.grad (for Hessian)
        gradient_phi= autograd.grad(loss_seq, self.net_phi.parameters(),
                                       create_graph=True, allow_unused=True)  # create_graph : for second derivative

        # gradient_phi_y = autograd.grad(loss_q_y, self.net_phi.parameters(),
        #                              create_graph=True, allow_unused=True)  # create_graph : for second derivative
        # gradient_phi_p = autograd.grad(loss_q_p, self.net_phi.parameters(),
        #                                create_graph=True, allow_unused=True)  # create_graph : for second derivative
        # gradient_phi_r = autograd.grad(loss_q_r, self.net_phi.parameters(),
        #                                create_graph=True, allow_unused=True)  # create_graph : for second derivative

        # return loss, gradient_phi, acc

        return loss_q_y, loss_q_p, loss_q_r, gradient_phi, q_y_pred, q_p_pred, q_r_pred


    def get_loss(self, y, p, r, label_y, label_cont):
        label_yaw = Variable(label_y[:, 0])
        label_pitch = Variable(label_y[:, 1])
        label_roll = Variable(label_y[:, 2])

        label_yaw_cont = Variable(label_cont[:, 0])
        label_pitch_cont = Variable(label_cont[:, 1])
        label_roll_cont = Variable(label_cont[:, 2])

        # Cross entropy loss
        loss_yaw = self.criterion(y.float(), label_yaw)
        loss_pitch = self.criterion(p.float(), label_pitch)
        loss_roll = self.criterion(r.float(), label_roll)

        # MSE loss
        yaw_predicted = self.softmax(y)
        pitch_predicted = self.softmax(p)
        roll_predicted = self.softmax(r)

        # print("predicted vals", loss_yaw, loss_pitch, loss_roll)

        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, 1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, 1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor, 1) * 3 - 99

        loss_reg_yaw = self.reg_criterion(yaw_predicted.float(), label_yaw_cont.float())
        loss_reg_pitch = self.reg_criterion(pitch_predicted.float(), label_pitch_cont.float())
        loss_reg_roll = self.reg_criterion(roll_predicted.float(), label_roll_cont.float())

        print('reg loss yaw', loss_reg_yaw)
        # # Total loss
        loss_yaw += self.alpha * loss_reg_yaw
        loss_pitch += self.alpha * loss_reg_pitch
        loss_roll += self.alpha * loss_reg_roll

        return loss_yaw, loss_pitch, loss_roll, yaw_predicted, pitch_predicted, roll_predicted
        # return loss_seq, grad_seq

    def net_forward(self, support_x, support_y, support_const):
        # theta update (general)
        # To write the merged gradients in net_theta network from metalearner

        y, p, r = self.net_theta(support_x, support_y)
        loss_y, loss_p, loss_r, y_pred, p_pred, r_pred = self.get_loss(y, p, r, support_x, support_const)

        return loss_y, loss_p, loss_r


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
        self.softmax = nn.Softmax()

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
        # dummy_loss.backward()
        grad_seq = [torch.tensor(1, dtype=float) for _ in range(len(dummy_loss))]
        torch.autograd.grad(dummy_loss, grad_seq)  # dummy_loss : summed gradients_phi (for general theta network)
        self.optimizer.step()

        for h in hooks:
            h.remove()

    def forward(self, support_x, support_y, support_cont, query_x, query_y, query_cont):
        # L[p]arned by Learner for every episode -> get the losses of parameter theta
        # Get loss and combine to update theta

        sum_grads_phi = None
        meta_batch_size = support_y.size(0)  # 5

        sum_grads_phi_y = None
        sum_grads_phi_p = None
        sum_grads_phi_r = None

        accs = []
        loss = []
        for i in range(meta_batch_size):

            loss_y, loss_p, loss_r, grad_phi, y,p,r = self.learner(support_x[i], support_y[i], support_cont[i],
                                                    query_x[i], query_y[i], query_cont[i],
                                                    self.num_updates)

            loss.append([loss_y, loss_p, loss_r])
            if sum_grads_phi is None:
                sum_grads_phi = grad_phi
            else:
                sum_grads_phi = [torch.add(i, j) for i, j in zip(sum_grads_phi, grad_phi) if i is not None]  # to get theta


            print(f'MAML Training ---> Yaw Loss: {abs(loss_y)}, Pitch Loss: {abs(loss_p)}, Roll Loss: {abs(loss_r)}')
            # return loss_y, loss_p, loss_r
        # loss_t_y, loss_t_p, loss_t_r = self.learner.net_forward(support_x[0], support_y[0], support_cont[0])
        # loss_seq = [loss_t_y, loss_t_p, loss_t_r]
        # self.meta_update(loss_seq, sum_grads_phi)

        # return loss_y, loss_p, loss_r

    def pred(self, support_x, support_y, support_cont, query_x, query_y, query_cont):
        meta_batch_size = support_y.size(0)
        total = 0
        yaw_error = .0
        pitch_error = .0
        roll_error = .0
        accs = []
        loss = []
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor)

        for i in range(meta_batch_size):
            loss_y, loss_p, loss_r, grad_out, pred_y, pred_p, pred_r = self.learner(support_x[i], support_y[i], support_cont[i], query_x[i], query_y[i], query_cont[i], self.num_updates)

            print(support_x[i][0].shape)
            img = utils.draw_axis(query_x[i][0], pred_y[i - 1], pred_p[i - 1], pred_r[i - 1])
            # plt.imshow(img[0])
            # plt.imshow(img[1])
            # plt.show()

            print(f'Testing Loss -----> yaw: {abs(loss_y)}, pitch: {abs(loss_p)}, roll: {abs(loss_r)}')

            # return loss_y, loss_p, loss_r, grad_out
            # loss.append((loss_y, loss_p, loss_r))

        # return loss