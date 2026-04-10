
import torch
import logging
import numpy as np
import copy
import os

from trainer import client

from torch.utils.data import DataLoader, Dataset
from util import get_dataset
from update import LocalUpdate
from AirComp_onebit import transmission,transmission_no_channel

class DatasetSplit(Dataset):
    def __init__(self, dataset, indxs):
        self.dataset = dataset
        self.indxs = list(indxs)

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, item):
        feature, target = self.dataset[int(self.indxs[item])]
        return feature, target

class Server():
    def __init__(self, model, dict_client, args, metrics, opt_params, channels):
        self.gmodel = model
        self.dict_client = dict_client
        self.opt_params = opt_params
        self.channels = channels

        self.args = args
        self.generator = torch.Generator()
        self.generator.manual_seed(self.args.seed)
        train_dataset, test_dataset, user_groups = get_dataset(args)
        self.trainData = train_dataset
        self.testData = test_dataset

        if dict_client == 1:
            self.trainloader = DataLoader(self.trainData, batch_size=self.args.b, shuffle=True, pin_memory=True)
            self.testloader = DataLoader(self.testData, batch_size=self.args.b, shuffle=False, pin_memory=True)
        else:
            trainloaders = {}
            testloaders = {}
            m = int(args.K)
            idxs_users = np.random.choice(range(args.K), m, replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                trainloaders[idx] = local_model.trainloader
                testloaders[idx] = local_model.testloader
            self.train_data_local_dict = trainloaders
            self.test_data_local_dict = testloaders
            self.trainloader = DataLoader(self.trainData, batch_size=self.args.b, shuffle=True, pin_memory=True)
            self.testloader = DataLoader(self.testData, batch_size=self.args.b, shuffle=False, pin_memory=True)

        self.metrics = metrics
        self.save_path = os.path.join(args.results_dir, args.save_dir)


    def train(self):
        acc_1 = 0.0
        args = self.args

        self.clients = []
        if self.dict_client:
            for i in range(self.args.K):
                dataloader = DataLoader(self.trainData, batch_size=self.args.b, shuffle=True, pin_memory=True)
                self.clients.append(client.Client(i, self.gmodel, dataloader))
        else:
            for i in range(self.args.K):
                dataloader = self.train_data_local_dict[i]
                self.clients.append(client.Client(i, self.gmodel, dataloader))

        for round in range(self.args.epochs):
            A, p, theta = self.opt_params[round]
            h_d, H_RB, h_UR = self.channels[round]
            np.random.seed(self.args.seed + round)
            torch.manual_seed(self.args.seed + round)
            ls_int = []
            if round == 0:
                trainacc, testacc, trainloss = self.accuracy(self.gmodel)
                logging.info("round: {}, train acc: {}, test acc: {}".format(round, trainacc, testacc))
                self.metrics.add(round=round, train_acc=trainacc, train_loss=trainloss, test_acc=testacc, local_steps=ls_int)
                self.metrics.save()

            logging.info("\n===> # %d", round+1)

            gradient = []
            paramNow = self.getParams(self.gmodel)
            selected_clients = list(range(self.args.K))
            for id in selected_clients:
                assigned_subband = np.where(A[id, :] == 1)[0][0]
                H_RB_id = H_RB[id, assigned_subband, :, :]
                h_UR_id = h_UR[assigned_subband, :, id]
                h_d_id = h_d[assigned_subband, :, id]
                h_id = np.dot(H_RB_id, np.diag(theta) @ h_UR_id) + h_d_id

                worker = self.clients[id]
                worker.reset(copy.deepcopy(paramNow))
                (net) = worker.train(self.args.lr, self.args.momentum, self.args.local_epoch, self.args.gpu,
                                                                    self.args.device, opti=self.args.local_opti,h=h_id,
                                                                    downlink=self.args.downlink,SNR_dl=self.args.SNR_dl)
                paraNew = self.getParams(net)
                param_delta = (paramNow - paraNew) / self.args.lr
                gradient.append(param_delta)

            paramChange = transmission(self.args, A, p, theta, H_RB, h_UR, h_d, gradient)
            if self.args.algorithm == 'onebit':
                paramNext = self.Onebit(paramChange)

            self.setParams(paramNext)
            logging.debug("round #%d aggregation finished!", round)

            if round % self.args.show == 0 or round >= self.args.rounds - 5:
                trainacc, testacc, trainloss = self.accuracy(self.gmodel)
                logging.info("round: {}, train acc: {}, test acc: {}".format(round+1, trainacc, testacc))
                self.metrics.add(round=round + 1, train_acc=trainacc, train_loss=trainloss, test_acc=testacc)
                self.metrics.save()

                if acc_1 < testacc:
                    acc_1 = testacc
        logging.info("best test acc: {}".format(acc_1))


    def accuracy(self, net):
        net.eval()
        train_acc = 0.0
        train_num = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_num = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        batch_loss = []
        with torch.no_grad():
            for i, (input, target) in enumerate(self.trainloader):
                if self.args.gpu==0:
                    input, target = input.to(self.args.device), target.to(self.args.device)
                    net = net.to(self.args.device)
                output = net(input)
                target = target.to(dtype=torch.long)
                loss = criterion(output, target)
                _, prediction = torch.max(output, 1)
                correct_num = prediction.eq(target).sum().item()
                train_num += target.size(0)
                train_acc += correct_num
                batch_loss.append(loss.item())
            train_loss = sum(batch_loss)/len(batch_loss)

            for i, (input, target) in enumerate(self.testloader):
                if self.args.gpu==0:
                    input, target = input.to(self.args.device), target.to(self.args.device)
                    net = net.to(self.args.device)
                output = net(input)
                _, prediction = torch.max(output, 1)
                correct_num = prediction.eq(target).sum().item()
                test_num += target.size(0)
                test_acc += correct_num

        return (train_acc / train_num), (test_acc / test_num), train_loss


    def Onebit(self, param_list):
        net = self.getParams(self.gmodel).to(self.args.device)
        temp = torch.zeros_like(net).to(self.args.device)
        for param in param_list:
            # param = param.to(self.args.device) # np.array cannot perform this
            param = torch.tensor(param).to(self.args.device)
            temp += param
        signs = torch.sign(temp)
        zero_indices = (signs == 0)
        if zero_indices.sum() > 0:
            random_signs = torch.randint(2, size=signs.shape,
                                         device=self.args.device) * 2 - 1  # generate {-1, 1} on the GPU
            random_signs = random_signs.float()
            signs[zero_indices] = random_signs[zero_indices]
        result = net - self.args.lr * signs # w = w - learning_rate * sign(gradient_global)
        return result

    def setParams(self, flat_params):
        prev_ind = 0
        with torch.no_grad():
            for param in self.gmodel.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.data.copy_(
                    flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
                prev_ind += flat_size

    def getParams(self, net):
        params = []
        for param in net.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params.detach().clone()