import copy
import torch
import numpy as np

import logging

class Client():
    def __init__(self, id, model, datasets):
        self.id = id
        self.model = copy.deepcopy(model)
        self.dataloader = datasets

    def reset(self, params):
        self.w0 = copy.deepcopy(params)
        self.setParams(params)

    def setParams(self, flat_params):
        prev_ind = 0
        with torch.no_grad():
            for param in self.model.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.data.copy_(
                    flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size

    def getParams(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.detach().clone()

    def noisydl_percsi(self, h, SNR_dl):
        net = self.getParams()
        temp = torch.zeros_like(net)
        sigma = ((torch.norm(net) / (10 ** (SNR_dl / 10))) ** 0.5) / (abs(h.real))
        noise = torch.normal(0.0, sigma, temp.shape)
        if torch.cuda.is_available():
            noise = noise.to("cuda:0")
        temp = net + noise
        return temp

    def noisydl_impercsi(self, h, h_est, SNR_dl):
        net = self.getParams()
        client_scale = (h * h_est.conjugate()) / (abs(h_est) ** 2)
        temp = torch.zeros_like(net)
        sigma = (torch.norm(net) / (10 ** (SNR_dl / 10)) ** 0.5) / (abs(h_est.real))
        noise = torch.normal(0.0, sigma, temp.shape)
        if torch.cuda.is_available():
            noise = noise.to("cuda:0")
        temp = client_scale * net + noise
        return temp

    def noisydl_impercsi_cifar(self, h, h_est, SNR_dl, dl_beta=50.0,
                               h_floor=1e-2, sigma_cap=0.05, eps=1e-8):
        net = self.getParams()
        h_abs = max(abs(h_est), h_floor)
        client_scale = ((h * h_est.conjugate()).real) / (abs(h_est) ** 2 + eps)
        net_rms = torch.norm(net) / (net.numel() ** 0.5)
        sigma = (net_rms * (10 ** (-SNR_dl / 20))) / (dl_beta * h_abs + eps)
        sigma = min(float(sigma), sigma_cap)
        noise = torch.normal(0.0, sigma, net.shape)
        if torch.cuda.is_available():
            noise = noise.to("cuda:0")
        temp = client_scale * net + noise
        return temp

    def train(self, lr, momentum, localEpoch, gpu, device, opti='fedavg', mu=0.1, lo_alg='AOAFL', alg='AOAFL', beta=0.5,
              pow=1, cs='yes', h=complex(1,0), h_hat=complex(1,0),alpha=0.2, downlink=1, SNR_dl=20,weightdecay=0,
              clipnorm=60, dataset='cifar10', dl_beta=50):

        if downlink == 1:
            param_dl = self.noisydl_percsi(h, SNR_dl)
            self.setParams(param_dl)
        elif downlink == 2:
            if dataset == 'cifar10':
                param_dl = self.noisydl_impercsi_cifar(h, h_hat, SNR_dl,dl_beta)
            else:
                param_dl = self.noisydl_impercsi(h, h_hat, SNR_dl)
            self.setParams(param_dl)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weightdecay)
        criterion = torch.nn.CrossEntropyLoss()
        steps = 0
        stop_sign = False
        total_loss = 0.0
        train_num = 0.0
        train_acc = 0.0
        for epoch in range(localEpoch):
            for i, (feature, target) in enumerate(self.dataloader):
                if gpu:
                    self.model = self.model.to(device)
                    feature, target = feature.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(feature)
                target = target.to(dtype=torch.long)
                loss = criterion(output, target)
                if opti == 'fedprox':
                    if i > 0:
                        w_diff = torch.pow(torch.linalg.norm(self.w0 - self.getParams()), 2)
                        loss += mu / 2.0 * w_diff
                elif (opti != 'fedavg') & (opti != 'fednova'):
                        raise ValueError("Local optimization method {} not supported".format(opti))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipnorm)
                optimizer.step()

                total_loss += loss.item()
                _, prediction = torch.max(output, 1)
                correct_num = prediction.eq(target).sum().item()
                train_num += target.size(0)
                train_acc += correct_num

                if (lo_alg == alg) & (alg=='AOAFL'):
                    if cs == 'yes':
                        steps += 1
                        delta = (beta * alpha * h_hat.conjugate()/ (steps * abs(h_hat)**2)) * (self.getParams() - self.w0)
                        if ((torch.norm(delta)) ** 2) <= pow:
                            stop_sign = True
                            break

            if stop_sign:
                break

        logging.info("client {}: local steps: {}, train loss: {}, train acc: {}".format(self.id, steps, loss / train_num, train_acc / train_num))
        return (self.model, steps)
