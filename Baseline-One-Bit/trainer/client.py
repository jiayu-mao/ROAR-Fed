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
        sigma = (torch.norm(net) / (10 ** (SNR_dl / 10)) ** 0.5) / (abs(h_est.real)) # update 1.27
        noise = torch.normal(0.0, sigma, temp.shape)
        if torch.cuda.is_available():
            noise = noise.to("cuda:0")
        temp = client_scale * net + noise
        return temp

    def noisydl(self, h, SNR_dl):
        net = self.getParams()
        client_scale = h.real
        client_scale = client_scale.item()
        temp = torch.zeros_like(net)
        sigma = (torch.norm(net) / (10 ** (SNR_dl / 10)) ** 0.5)
        noise = torch.normal(0.0, sigma, temp.shape)
        if torch.cuda.is_available():
            noise = noise.to("cuda:0")
        temp = client_scale * net + noise
        return temp

    def train(self, lr, momentum, localEpoch, gpu, device, opti='fedavg', mu=0.1, h=complex(1,0), h_hat=complex(1,0), downlink=1, SNR_dl=20):
        if downlink >0:
            param_dl = self.noisydl(h, SNR_dl)
            self.setParams(param_dl)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,weight_decay=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        train_num = 0.0
        train_acc = 0.0
        if gpu == 0:
            self.model = self.model.to(device)
        for epoch in range(localEpoch):
            for i, (feature, target) in enumerate(self.dataloader):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                optimizer.step()

                total_loss += loss.item()
                _, prediction = torch.max(output, 1)
                correct_num = prediction.eq(target).sum().item()
                train_num += target.size(0)
                train_acc += correct_num

        logging.info("client {}: train loss: {}, train acc: {}".format(self.id,  loss / train_num, train_acc / train_num))
        return (self.model)