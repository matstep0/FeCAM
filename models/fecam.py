
import logging
import numpy as np
from tqdm import tqdm
import torch

from torch import nn
from torch import optim
from torch import linalg as LA
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
from utils.maha_utils import compute_common_cov, compute_new_common_cov, compute_new_cov
from sklearn import svm
from collections import namedtuple
from sklearn.covariance import EllipticEnvelope
import itertools
from sklearn.ensemble import IsolationForest

EPSILON = 1e-8


class FeCAM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = CosineIncrementalNet(args, False)
        self._protos = []
        self._init_protos = []
        self._common_cov = None
        self._cov_mat = []
        self._diag_mat = []
        self._common_cov_shrink = None
        self._cov_mat_shrink = []
        self._norm_cov_mat = []
        self._ocsvm_models = {}
        self._elliptic_envelopes = {}
        self._isolation_forests = {}

    def after_task(self):
        self._known_classes = self._total_classes
        # if self._cur_task == 0:
        #     self.save_checkpoint("{}_{}_{}_{}".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"]))
        
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]

        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:   # Freezing the network
            for p in self._network.convnet.parameters():
                p.requires_grad = False
        
        self.shot = None

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', shot=self.shot)  

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        resume = True  # set resume=True to use saved checkpoints after first task
        if self._cur_task == 0:
            if resume:
                self._network.load_state_dict(torch.load("{}_{}_{}_{}_{}.pkl".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                self._epoch_num = self.args["init_epochs"]
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.args["init_epochs"])
                self._train_function(train_loader, test_loader, optimizer, scheduler)
            self._build_base_protos()
            self._build_protos()
            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:  # we apply covariance shrinkage 2 times to obtain better estimates of matrices
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))
        else:
            self._cov_mat_shrink, self._norm_cov_mat, self._diag_mat = [], [], []
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            self._build_protos()
            self._update_fc()
            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_new_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))
        
        vectors, y_true = self._extract_vectors(train_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        classes = np.unique(y_true)
        class_to_data = {cls: [] for cls in classes}

        for vector, label in zip(vectors, y_true):
            class_to_data[label].append(vector)

        # ONE_CLASS SVM

        # print('TRAINING ONE CLASS SVM')

        # accuracies = []
        # gamma = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 30, 50, 100]
        # nu = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        # kernel = 'rbf'

        # for g, n in itertools.product(gamma, nu):
        #     for cls, data in class_to_data.items():
        #         model = svm.OneClassSVM(gamma=g, nu=n, kernel=kernel).fit(data)
        #         self._ocsvm_models[cls] = model
        #     _, _, acc = self.eval_task()
        #     accuracies.append(acc['top1'])
        #     print(f'gamma: {g}, nu: {n}, kernel: {kernel}, accuracy: {acc["top1"]}')
        
        # best_acc_idx = torch.argmax(torch.tensor(accuracies)).item()
        # best_gamma, best_nu = list(itertools.product(gamma, nu))[best_acc_idx]
        # print(f'OCSVM GRID task: {self._cur_task}, accuracy: {accuracies[best_acc_idx]}, gamma: {best_gamma}, nu: {best_nu}, kernel: {kernel}')

        # for cls, data in class_to_data.items():
        #     model = svm.OneClassSVM(gamma=best_gamma, nu=best_nu, kernel=kernel).fit(data)
        #     self._ocsvm_models[cls] = model

        # ELLIPTIC ENVELOPE

        # print('TRAINING ELLIPTIC ENVELOPE')

        # support_fraction = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        # contamination = [0.001, 0.01, 0.1, 0.2]
        # accuracies = []

        # for (sf, c) in itertools.product(support_fraction, contamination):
        #     for cls, data in class_to_data.items():
        #         model = EllipticEnvelope(random_state=0, support_fraction=sf, contamination=c).fit(data)
        #         self._elliptic_envelopes[cls] = model
        #     _, _, acc = self.eval_task()
        #     accuracies.append(acc['top1'])
        #     print(f'support_fraction: {sf}, contamination: {c}, accuracy: {acc["top1"]}')

        # best_acc_idx = torch.argmax(torch.tensor(accuracies)).item()
        # best_params = list(itertools.product(support_fraction, contamination))[best_acc_idx]
        # best_sf, best_c = best_params
        # print(f'ELLIPTIC ENVELOPE GRID task: {self._cur_task}, accuracy: {accuracies[best_acc_idx]}, support_fraction: {best_sf}, contamination: {best_c}')

        # for cls, data in class_to_data.items():
        #     model = EllipticEnvelope(random_state=0, support_fraction=best_sf, contamination=best_c).fit(data)
        #     self._elliptic_envelopes[cls] = model

        # ISOLATION FOREST

        print('TRAINING ISOLATION FOREST')
        n_estimators = [100, 200, 300]
        contamination = [0.001, 0.01, 0.1, 0.2]
        max_features = [1, 2, 3, 5]
        accuracies = []

        for (ne, c, mf) in itertools.product(n_estimators, contamination, max_features):
            for cls, data in class_to_data.items():
                model = IsolationForest(random_state=0, n_estimators=ne, contamination=c, max_features=mf).fit(data)
                self._isolation_forests[cls] = model
            _, _, acc = self.eval_task()
            accuracies.append(acc['top1'])
            print(f'n_estimators: {ne}, contamination: {c}, max_features: {mf}, accuracy: {acc["top1"]}')
        
        best_acc_idx = torch.argmax(torch.tensor(accuracies)).item()
        best_params = list(itertools.product(n_estimators, contamination, max_features))[best_acc_idx]
        best_ne, best_c, best_mf = best_params
        print(f'ISOLATION FOREST GRID task: {self._cur_task}, accuracy: {accuracies[best_acc_idx]}, \
                n_estimators: {best_ne}, contamination: {best_c}, max_features: {best_mf}')

        for cls, data in class_to_data.items():
            model = IsolationForest(random_state=0, n_estimators=best_ne, contamination=best_c, max_features=best_mf).fit(data)
            self._isolation_forests[cls] = model


    def _build_base_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            class_mean = self._network.fc.weight.data[class_idx]
            self._init_protos.append(class_mean)

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', shot=self.shot, ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(torch.tensor(class_mean).to(self._device))

    def _update_fc(self):
        self._network.fc.fc2.weight.data = torch.stack(self._protos[-self.args["increment"]:], dim=0).to(self._device)  # for cosine incremental fc layer

    
    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            if self._cur_task == 0:
                self._network.train()
            else:
                self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._cur_task ==0:
                    logits = self._network(inputs)['logits']
                else:
                    logits = self._network_module_ptr.fc(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
