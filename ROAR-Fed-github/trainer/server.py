import torch
import logging
import numpy as np
import copy
import os

from trainer import client
from RIS import phase_sca

from torch.utils.data import DataLoader, Dataset
from util import get_dataset, get_lr, get_lr_schedule
from update import LocalUpdate

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
    def __init__(self, model, traindata, testdata, dict_client, args, metrics):
        self.gmodel = model
        self.trainData = traindata
        self.testData = testdata
        self.dict_client = dict_client
        self.args = args
        self.generator = torch.Generator()
        self.generator.manual_seed(self.args.seed)
        train_dataset, test_dataset, user_groups = get_dataset(args)
        if dict_client == 1:
            self.trainloader = DataLoader(traindata, batch_size=self.args.b, shuffle=True, generator=self.generator, pin_memory=True)
            self.testloader = DataLoader(testdata, batch_size=self.args.b, shuffle=False, pin_memory=True)
        else:
            trainloaders = {}
            testloaders = {}
            m = int(args.clients_perc * args.total_clients)
            idxs_users = np.random.choice(range(args.total_clients), m, replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                trainloaders[idx] = local_model.trainloader
                testloaders[idx] = local_model.testloader
            self.train_data_local_dict = trainloaders
            self.test_data_local_dict = testloaders
            self.trainloader = DataLoader(traindata, batch_size=self.args.b, shuffle=True, generator=self.generator, pin_memory=True)
            self.testloader = DataLoader(testdata, batch_size=self.args.b, shuffle=False, pin_memory=True)

        self.metrics = metrics
        self.save_path = os.path.join(args.results_dir, args.save_dir)
        self.lr_schedule = get_lr_schedule(
            self.args.dataset,
            self.args.downlink,
            self.args.diri_beta
        )


    def train(self):
        acc_1 = 0.0
        args = self.args
        ref = args.ref

        self.clients = []
        if self.dict_client:
            for i in range(self.args.total_clients):
                dataloader = DataLoader(self.trainData, batch_size=self.args.b, shuffle=True, generator=self.generator, pin_memory=True)
                self.clients.append(client.Client(i, self.gmodel, dataloader))
        else:
            for i in range(self.args.total_clients):
                dataloader = self.train_data_local_dict[i]
                self.clients.append(client.Client(i, self.gmodel, dataloader))


        for round in range(self.args.rounds):
            self.args.lr = get_lr(round, self.lr_schedule)
            np.random.seed(self.args.seed + round)
            torch.manual_seed(self.args.seed + round)

            ls_int = []
            if round == 0:
                trainacc, testacc, trainloss = self.accuracy(self.gmodel)
                logging.info("round: {}, train acc: {}, test acc: {}".format(round, trainacc, testacc))
                self.metrics.add(round=round, train_acc=trainacc, train_loss=trainloss, test_acc=testacc, local_steps=ls_int)
                self.metrics.save()

            logging.info("\n===> # %d", round+1)
            if self.args.clients_perc == 1.0:
                selected_clients = list(range(self.args.total_clients))

            logging.info("clients: {}".format(list(selected_clients)))

            h_d = (np.random.randn(args.N, args.M) + 1j * np.random.randn(args.N, args.M)) / 2 ** 0.5
            h_d = h_d @ np.diag(args.PL_direct ** 0.5) / ref # direct link!
            H_RB = (np.random.randn(args.N, args.L) + 1j * np.random.randn(args.N, args.L)) / 2 ** 0.5
            h_UR = (np.random.randn(args.L, args.M) + 1j * np.random.randn(args.L, args.M)) / 2 ** 0.5
            h_UR = h_UR @ np.diag(args.PL_RIS ** 0.5) / ref
            H_RB_noise = (args.cs_sigma_hat * (np.random.randn(args.N, args.L) + 1j * np.random.randn(args.N, args.L))) / 2 ** 0.5
            h_UR_noise = (np.random.randn(args.L, args.M) + 1j * np.random.randn(args.L, args.M)) / 2 ** 0.5
            h_UR_noise = h_UR_noise @ np.diag((args.cs_sigma_hat * args.PL_RIS) ** 0.5) / ref
            h_d_noise = (np.random.randn(args.N, args.M) + 1j * np.random.randn(args.N, args.M)) / 2 ** 0.5
            h_d_noise = h_d_noise @ np.diag((args.cs_sigma_hat * args.PL_direct) ** 0.5) / ref

            G = np.zeros([args.N, args.L, args.M], dtype=complex)
            if self.args.csi == 1:
                for j in range(args.M):
                    G[:, :, j] = H_RB @ np.diag(h_UR[:, j])
            else:
                H_RB_est = H_RB + H_RB_noise
                h_UR_est = h_UR + h_UR_noise
                for j in range(args.M):
                    G[:, :, j] = H_RB_est @ np.diag(h_UR_est[:, j])

            if self.args.RIS_num == 1:
                theta = np.ones([args.L], dtype=complex)
                if self.args.phase_design == 'sca':
                    if round == 0:
                        theta_update = theta
                    else:
                        SGD_bound = 1
                        alpha = 1/self.args.M
                        const = self.args.lr * 1 * self.args.s_beta * alpha * SGD_bound * 20 / self.args.P
                        const = 15
                        s_t = const - h_d
                        mu_sca=200
                        iter_sca = 50
                        theta_update = phase_sca(self.args.L, theta, ls, s_t, G, mu_sca, iter_sca)

            paramChange = []
            ls = []
            alpha_record = []
            paramNow = self.getParams(self.gmodel)
            h = np.zeros([args.M], dtype=complex)
            for id in selected_clients:
                worker = self.clients[id]
                worker.reset(copy.deepcopy(paramNow))
                total_samples = len(worker.dataloader.dataset)
                alpha = total_samples / 60000
                if self.args.client_scale == 'yes':
                    if self.args.RIS_num == 1:
                        h = h_d[:,id] + G[:, :, id] @ theta_update
                    h = h[0]
                    cs_h_real = h.real
                    cs_h_img = h.imag
                    cs_std = torch.tensor(1.0)
                    cs_h_real = torch.tensor(cs_h_real)
                    cs_h_img = torch.tensor(cs_h_img)
                    cs_noise_real = h_d_noise[:,id].real
                    cs_noise_img = h_d_noise[:,id].imag
                    cs_noise_real = torch.tensor(cs_noise_real)
                    cs_noise_img = torch.tensor(cs_noise_img)

                    if torch.cuda.is_available():
                        cs_h_real = cs_h_real.to("cuda:0")
                        cs_h_img = cs_h_img.to("cuda:0")
                        cs_noise_real = cs_noise_real.to("cuda:0")
                        cs_noise_img = cs_noise_img.to("cuda:0")
                    cs_h = complex(cs_h_real, cs_h_img)
                    cs_noise = complex(cs_noise_real, cs_noise_img)
                    if self.args.csi == 1:
                        cs_h_hat = cs_h
                    else:
                        cs_h_hat = cs_h + cs_noise

                    (net, local_steps) = worker.train(self.args.lr, self.args.momentum, self.args.local_epoch, self.args.gpu,
                                                                self.args.device, opti=self.args.local_opti,
                                                                mu=self.args.mu, lo_alg=self.args.local_alg,
                                                                alg=self.args.algorithm, beta=self.args.s_beta,
                                                                pow=self.args.P, cs=self.args.client_scale,
                                                                h=cs_h,h_hat=cs_h_hat,alpha=alpha,downlink=self.args.downlink,
                                                                SNR_dl=self.args.SNR_dl,weightdecay=self.args.weightdecay,
                                                                clipnorm=self.args.clip_norm, dataset=self.args.dataset, dl_beta=self.args.dl_beta)
                paraNew = self.getParams(net)

                if self.args.client_scale == 'yes':
                    client_scale = (cs_h * cs_h_hat.conjugate()) / (abs(cs_h_hat) ** 2)
                    param_delta = client_scale.real * (paraNew - paramNow)

                paramChange.append(param_delta)
                ls.append(local_steps)
                alpha_record.append(alpha)

            if self.args.algorithm == 'AOAFL':
                paramNext = self.AOAFL(paramChange, ls, alpha_record)

            self.setParams(paramNext)
            logging.debug("round #%d aggregation finished!", round)

            if round % self.args.show == 0 or round >= self.args.rounds - 5:
                trainacc, testacc, trainloss = self.accuracy(self.gmodel)
                logging.info("round: {}, train acc: {}, test acc: {}".format(round+1, trainacc, testacc))
                self.metrics.add(round=round + 1, train_acc=trainacc, train_loss=trainloss, test_acc=testacc, local_steps=ls)
                self.metrics.plot(message=['train_acc', 'test_acc'], title='Accuracy', x_axis_label='Round', y_axis_label='Accuracy')
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
                if self.args.gpu:
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
                if self.args.gpu:
                    input, target = input.to(self.args.device), target.to(self.args.device)
                    net = net.to(self.args.device)
                output = net(input)
                _, prediction = torch.max(output, 1)
                correct_num = prediction.eq(target).sum().item()
                test_num += target.size(0)
                test_acc += correct_num

        return (train_acc / train_num), (test_acc / test_num), train_loss


    def AOAFL(self, param_list, steps,alpha_record):
        net = self.getParams(self.gmodel)
        temp = torch.zeros_like(net)
        i=0
        beta_client = []
        for param in param_list:
            alpha = alpha_record[i]
            betai = alpha / steps[i]
            beta_client.append(betai)
            temp += betai * param
            i =i+1
        sigma = (1 / self.args.s_beta) * ( (1 / (10 ** (self.args.SNR / 10))) ** 0.5)
        noise = torch.normal(0.0, sigma, temp.shape)
        if torch.cuda.is_available():
            noise = noise.to("cuda:0")
        temp = net + self.args.glr * temp + noise
        return temp


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