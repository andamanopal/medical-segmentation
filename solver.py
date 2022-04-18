import os
import time
import numpy as np
from datetime import datetime
import joblib

import torch
from torch import nn, optim
from torch.backends import cudnn
from tensorboardX import SummaryWriter
import optuna

import matplotlib.pyplot as plt
from torchvision import transforms as T

import data_loader
import misc, metric, random

import my_network.network as network
from my_network.UNet import UNet
from my_network.CustomUNet import CustomUNet
from my_network.UNetMultiConnection import UNetMultiConnection
from my_network.DoubleUNet import DoubleUNet
from my_network.ResUNet import ResUNet
from my_network.ResUNetMultiConnection import ResUNetMultiConnection

# seed = 20180945
# print(f'Using Seed : {seed}')
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)


class Solver(object):
    def __init__(self, config):
        # Get configuration parameters
        self.config = config

        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()

        # Model, optimizer, criterion
        self.models, self.optimizers, self.criteria = dict(), dict(), dict()

        # Loss, metric
        self.loss, self.metric = dict(), dict()

        # Training status
        self.phase_types = ["train", "valid"]
        self.lr = self.config.lr_opt["init"]
        self.init_lr_config = self.config.lr_opt
        self.complete_epochs = 0
        self.best_metric, self.best_epoch = 0, 0

        # Model and loss types
        self.model_types, self.optimizer_types = list(), list()
        self.loss_types, self.metric_types = list(), list()
        self.config.activation = getattr(nn, self.config.activation)

        # Member variables for data
        self.images, self.labels, self.weights, self.outputs = None, None, None, None
        self.model_types = [self.config.model_type]
        self.optimizer_types = [self.config.model_type]
        self.loss_types = ["bce", "l1"]
        self.metric_types = ["dice"]

        # Tensorboard
        self.tuning_log = list()
        self.tensorboard = None
        if self.config.phase in ['tune', 'tuning']:
            assert self.config.tuning_log_dir is not None, "Please specify tuning_log_dir"
            assert self.config.tuning_log_dir.endswith('/'), 'tuning_log_dir must ends with forward slash "/" '
            self.tuning_log_path = f"{self.config.model_type}_{str(datetime.now())[:-7].replace(':', '-')}"
            self.tuning_log_path = os.path.join(self.config.tuning_log_dir, self.tuning_log_path)

        # Check if device ids are string
        if isinstance(self.config.device_ids[0], str):
            self.config.device_ids = list(map(lambda x: int(x), self.config.device_ids))

        # CPU or CUDA
        self.device = torch.device("cuda:%d" % self.config.device_ids[0] if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True if torch.cuda.is_available() else False

        # Get data loader & build new model or load existing one
        self.load_model()
        self.get_data_loader(data_loader.get_loader)

    def get_data_loader(self, image_loader):
        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()
        for phase in ["train", "valid", "test"]:
            self.image_loader[phase] = image_loader(dataset_path=self.config.dataset_path,
                                                    num_classes=self.config.num_classes,
                                                    phase=phase,
                                                    shuffle=True,
                                                    patch_size=self.config.patch_size,
                                                    sample_weight=self.config.sample_weight,
                                                    batch_size=self.config.batch_size if phase in ["train",
                                                                                                   "valid"] else 4,
                                                    num_workers=self.config.num_workers)
            self.num_images[phase] = int(self.image_loader[phase].dataset.__len__())
            self.num_steps[phase] = int(np.ceil(self.num_images[phase] /
                                                (self.config.batch_size if phase in ["train", "valid"] else 4)))

    def build_model(self):
        # Build model

        if self.config.model_type == "UNet":
            self.models[self.model_types[0]] = UNet(in_channels=self.config.num_img_ch,
                                                    out_channels=self.config.num_classes,
                                                    num_features=self.config.num_features)
        elif self.config.model_type == "CustomUNet":
            self.models[self.model_types[0]] = CustomUNet(in_channels=self.config.num_img_ch,
                                                          out_channels=self.config.num_classes,
                                                          num_features=self.config.num_features,
                                                          activation=self.config.activation)
        elif self.config.model_type == "UNetMultiConnection":
            self.models[self.model_types[0]] = UNetMultiConnection(in_channels=self.config.num_img_ch,
                                                                   out_channels=self.config.num_classes,
                                                                   num_features=self.config.num_features,
                                                                   activation=self.config.activation)
        elif self.config.model_type == "DoubleUNet":
            self.models[self.model_types[0]] = DoubleUNet(in_channels=1, freeze_vgg=False)
        elif self.config.model_type == "ResUNet":
            self.models[self.model_types[0]] = ResUNet(in_channels=self.config.num_img_ch,
                                                       out_channels=self.config.num_classes,
                                                       num_features=self.config.num_features,
                                                       activation=self.config.activation,
                                                       resnet_type=self.config.resnet_type,
                                                       weight_path=self.config.weight_path)
        elif self.config.model_type == "ResUNetMultiConnection":
            self.models[self.model_types[0]] = ResUNetMultiConnection(in_channels=self.config.num_img_ch,
                                                                      out_channels=self.config.num_classes,
                                                                      num_features=self.config.num_features,
                                                                      activation=self.config.activation, )
        else:
            raise NotImplementedError("Model type [%s] is not implemented" % self.config.model_type)
        # Build optimizer
        if self.config.optimizer is None:
            self.optimizers[self.model_types[0]] = optim.Adam(self.models[self.model_types[0]].parameters(),
                                                              lr=self.config.lr_opt["init"],
                                                              betas=(0.9, 0.999),
                                                              weight_decay=self.config.l2_penalty)
        else:
            self.optimizers[self.model_types[0]] = (
                getattr(optim, self.config.optimizer)(self.models[self.model_types[0]].parameters(),
                                                      lr=self.config.lr_opt["init"],
                                                      weight_decay=self.config.l2_penalty))

        # Build criterion
        self.criteria["bce"] = nn.BCELoss  # Binary cross entropy
        self.criteria["l1"] = nn.L1Loss()  # absolute-value norm (L1 norm)

        # Model initialization
        for model_type in self.model_types:
            self.models[model_type] = network.init_net(self.models[model_type],
                                                       init_type="kaiming", init_gain=0.02,
                                                       device_ids=self.config.device_ids)

    def save_model(self, epoch):
        checkpoint = {"config": self.config,
                      "lr": self.lr,
                      "model_types": self.model_types,
                      "optimizer_types": self.optimizer_types,
                      "loss_types": self.loss_types,
                      "complete_epochs": epoch + 1,
                      "best_metric": self.best_metric,
                      "best_epoch": self.best_epoch}
        model_state_dicts = {"model_%s_state_dict" % model_type:
                                 self.models[model_type].state_dict() for model_type in self.model_types}
        optimizer_state_dicts = {"optimizer_%s_state_dict" % optimizer_type:
                                     self.optimizers[optimizer_type].state_dict() for optimizer_type in
                                 self.optimizer_types}
        checkpoint = dict(checkpoint, **model_state_dicts)
        checkpoint = dict(checkpoint, **optimizer_state_dicts)
        torch.save(checkpoint, os.path.join(self.config.model_path, "model.pth"))

        print("Best model (%.3f) is saved to %s" % (self.best_metric, self.config.model_path))

    def save_epoch(self, epoch):
        temp = torch.load(os.path.join(self.config.model_path, "model.pth"))
        temp["lr"] = self.lr
        temp["complete_epochs"] = epoch + 1
        torch.save(temp, os.path.join(self.config.model_path, "model.pth"))

        print("")

    def load_model(self):
        if os.path.isfile(os.path.join(self.config.model_path, "model.pth")):
            checkpoint = torch.load(os.path.join(self.config.model_path, self.config.model_name))

            self.config = checkpoint["config"]
            self.lr = checkpoint["lr"]
            self.model_types = checkpoint["model_types"]
            self.optimizer_types = checkpoint["optimizer_types"]
            self.loss_types = checkpoint["loss_types"]
            self.complete_epochs = checkpoint["complete_epochs"]
            self.best_metric = checkpoint["best_metric"]
            self.best_epoch = checkpoint["best_epoch"]

            self.build_model()
            self.load_model_state_dict(checkpoint)
        else:
            self.build_model()

    def load_model_state_dict(self, checkpoint):
        for model_type in self.model_types:
            self.models[model_type].load_state_dict(checkpoint["model_%s_state_dict" % model_type])
        for optimizer_type in self.optimizer_types:
            self.optimizers[optimizer_type].load_state_dict(checkpoint["optimizer_%s_state_dict" % optimizer_type])

    def set_train(self, is_train=True):
        for model_type in self.model_types:
            if is_train:
                self.models[model_type].train(True)
            else:
                self.models[model_type].eval()

    def update_lr(self, epoch, improved=False):
        if self.config.lr_opt["policy"] == "linear":
            self.lr = self.config.lr_opt["init"] / (1.0 + self.config.lr_opt["gamma"] * epoch)
        elif self.config.lr_opt["policy"] == "flat_linear":
            self.lr = self.config.lr_opt["init"]
            if epoch > self.config.lr_opt["step_size"]:
                self.lr /= (1.0 + self.config.lr_opt["gamma"] * (epoch - self.config.lr_opt["step_size"]))
        elif self.config.lr_opt["policy"] == "step":
            self.lr = self.config.lr_opt["init"] * self.config.lr_opt["gamma"] ** \
                      int(epoch / self.config.lr_opt["step_size"])
        elif self.config.lr_opt["policy"] == "plateau":
            if not improved:
                self.config.lr_opt["step"] += 1
                if self.config.lr_opt["step"] >= self.config.lr_opt["step_size"]:
                    self.lr *= self.config.lr_opt["gamma"]
                    self.config.lr_opt["step"] = 0
            else:
                self.config.lr_opt["step"] = 0
        else:
            return NotImplementedError("Learning rate policy [%s] is not implemented", self.config.lr_opt["policy"])

        for optimizer_type in self.optimizer_types:
            for param_group in self.optimizers[optimizer_type].param_groups:
                param_group["lr"] = self.lr

        ending = False if self.lr >= self.config.lr_opt["term"] else True
        return ending

    def print_info(self, phase="train", print_func=None, epoch=0, step=0):
        # Assert
        assert (phase in self.phase_types)

        # Print process information
        total_epoch = self.complete_epochs + self.config.num_epochs
        total_step = self.num_steps[phase]

        prefix = "[Epoch %4d / %4d] lr %.1e" % (epoch, total_epoch, self.lr)
        suffix = "[%s] " % phase
        for loss_type in self.loss_types:
            suffix += "%s: %.5f / " % (loss_type,
                                       sum(self.loss[loss_type][phase]) / max([len(self.loss[loss_type][phase]), 1]))
        for metric_type in self.metric_types:
            suffix += "%s: %.5f / " % (metric_type,
                                       sum(self.metric[metric_type][phase]) / max(
                                           [len(self.metric[metric_type][phase]), 1]))
        if print_func is not None:
            print_func(step + 1, total_step, prefix=prefix, suffix=suffix, dec=2, bar_len=30)
        else:
            print(suffix, end="")

    def log_to_tensorboard(self, epoch, elapsed_time=None, intermediate_output=None, accuracy=None):
        if elapsed_time is not None:
            self.tensorboard.add_scalar("elapsed_time", elapsed_time, epoch)
        self.tensorboard.add_scalar("learning_rate", self.lr, epoch)
        for loss_type in self.loss_types:
            self.tensorboard.add_scalars("%s" % loss_type, {phase: sum(self.loss[loss_type][phase]) /
                                                                   max([len(self.loss[loss_type][phase]), 1])
                                                            for phase in self.phase_types}, epoch)
        for metric_type in self.metric_types:
            self.tensorboard.add_scalars("%s" % metric_type, {phase: sum(self.metric[metric_type][phase]) /
                                                                     max([len(self.metric[metric_type][phase]), 1])
                                                              for phase in self.phase_types}, epoch)
        if (epoch % 10) == 0:
            if intermediate_output is not None:
                self.tensorboard.add_image("intermediate_output", intermediate_output, epoch)
            if accuracy is not None:
                self.tensorboard.add_scalars("accuracy", {"f-score": accuracy[0],
                                                          "precision": accuracy[1],
                                                          "recall": accuracy[2]}, epoch)

    def forward(self, images, labels, weights):
        # Image to device
        self.images = images.to(self.device)  # n1hw (grayscale)
        self.labels = labels.to(self.device)  # n2hw (binary classification)
        self.weights = weights.to(self.device)  # n1hw?

        # Prediction (forward)
        self.outputs = self.models[self.model_types[0]](self.images)

    def backward(self, phase="train"):
        # Backward to calculate the gradient
        # Loss defition
        bce_loss = self.criteria["bce"](self.weights)(self.outputs, self.labels)
        l1_loss = self.config.l1_weight * self.criteria["l1"](self.outputs, self.labels)

        # Loss integration and gradient calculation (backward)
        loss = bce_loss + l1_loss
        if phase == "train":
            loss.backward()

        self.loss["bce"][phase].append(bce_loss.item())
        self.loss["l1"][phase].append(l1_loss.item())

    def optimize(self, backward):
        """ Optimize and update weights according to the calculated gradients. """
        self.optimizers[self.optimizer_types[0]].zero_grad()
        backward()
        self.optimizers[self.optimizer_types[0]].step()

    def calculate_metric(self, phase="train"):
        if phase == 'train':
            self.metric["dice"][phase].append(metric.get_similiarity(self.outputs, self.labels, ch=1))
        elif phase == 'valid':
            self.metric["dice"][phase].append(
                metric.get_similiarity(torch.where(self.outputs > 0.5, 1, 0), self.labels, ch=1))

    def train(self, tuning_trial=None, params=None):
        self.tensorboard = SummaryWriter(os.path.join(self.config.model_path, "logs"))
        print(
            f"Trainable Parameters : {sum(p.numel() for p in self.models[self.model_types[0]].parameters() if p.requires_grad)}")
        for epoch in range(self.complete_epochs, self.complete_epochs + self.config.num_epochs):
            # ============================= Training ============================= #
            # ==================================================================== #

            # Training status parameters
            t0 = time.time()
            self.loss = {loss_type: {"train": list(), "valid": list()} for loss_type in self.loss_types}
            self.metric = {metric_type: {"train": list(), "valid": list()} for metric_type in self.metric_types}

            # Image generating for training process
            self.set_train(is_train=True)
            for i, (images, labels, weights) in enumerate(self.image_loader["train"]):
                # Forward
                self.forward(images, labels, weights)

                # Backward & Optimize
                self.optimize(self.backward)

                # Calculate evaluation metrics
                self.calculate_metric()

                # Print training info
                self.print_info(phase="train", print_func=misc.print_progress_bar,
                                epoch=epoch + 1, step=i)

            # ============================ Validation ============================ #
            # ==================================================================== #
            # Image generating for validation process
            with torch.no_grad():
                self.set_train(is_train=False)
                for i, (images, labels, weights) in enumerate(self.image_loader["valid"]):
                    # Forward
                    self.forward(images, labels, weights)

                    # Backward
                    self.backward(phase="valid")

                    # Calculate evaluation metrics
                    self.calculate_metric(phase="valid")

            # Print validation info
            self.print_info(phase="valid")

            # Check pruning condition for Optuna
            if tuning_trial is not None:
                tuning_trial.report(np.mean(list(map(lambda x: x.item(), self.metric["dice"]["valid"]))), epoch)
                if tuning_trial.should_prune():
                    print("Pruning the trial....")
                    self.load_model()
                    score = self.test()
                    self.tuning_log += [dict(params=params, score=score)]
                    with open(f'{self.tuning_log_path}.txt', 'a+') as f:
                        for trial in self.tuning_log:
                            line = str(trial['params']) + '  ' + str(trial['score'])
                            f.write(line)
                            f.write('\n')
                    raise optuna.TrialPruned()

            # Tensorboard logs
            self.log_to_tensorboard(epoch + 1, elapsed_time=time.time() - t0)

            # ============================ Model Save ============================ #
            # ==================================================================== #
            # Best valiation metric logging
            valid_metric = (sum(self.metric["dice"]["valid"]) / len(self.metric["dice"]["valid"])).item()
            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                self.best_epoch = epoch + 1

                # Model save
                self.save_model(epoch)
            else:
                # Save current epoch
                self.save_epoch(epoch)

            # Learning rate adjustment
            if self.update_lr(epoch, epoch == (self.best_epoch - 1)):
                print("Model is likely to be fully optimized. Terminating the training...")
                break

        if self.config.phase != 'tuning':
            self.tensorboard.close()

    def test(self):
        # Image generating for test process
        self.set_train(is_train=False)

        score = {'precision': [], 'recall': [], 'f1': [], 'dice': []}

        with torch.no_grad():
            for i, (images, labels, weights) in enumerate(self.image_loader["test"]):
                # Image to device
                images = images.to(self.device)  # n1hw (grayscale)

                # Make prediction
                outputs = self.models[self.model_types[0]](images).detach().cpu()

                for index, pred in enumerate(outputs):
                    uint8_label = (labels[index][1].numpy() * 255).astype(np.uint8)
                    th_output = np.where(pred[1].numpy() > 0.5, 255, 0).astype(np.uint8)
                    prec, rec, f1 = metric.calculate_precision_recall_f1(th_output, uint8_label)
                    score['precision'] += [prec]
                    score['recall'] += [rec]
                    score['f1'] += [f1]
                    score['dice'] += [metric.get_similiarity(T.ToTensor()(th_output), labels[index][1])]
                    # score['dice'] += [metric.get_similiarity(pred[1], labels[index][1])]

        score['precision'] = np.mean(score['precision'])
        score['recall'] = np.mean(score['recall'])
        score['f1'] = np.mean(score['f1'])
        score['dice'] = np.mean(score['dice'])
        print(score)
        return score

    def objective(self, trial):

        params = {
            'num_features': trial.suggest_int('num_features', 32, 96),
            # 'kernel_size': trial.suggest_int('kernel_size', 2, 7),
            'activation': trial.suggest_categorical("activation", ["ReLU", "PReLU", "LeakyReLU"]),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
            'l2_penalty': trial.suggest_loguniform("l2_penalty", 1e-4, 1e-1)
        }
        print('Using Parameters ...')
        print(params, '\n')

        # Change model's config
        self.config.num_features = params['num_features']
        self.config.activation = getattr(nn, params['activation'])
        print(self.config.activation)
        self.config.l2_penalty = params['l2_penalty']
        self.build_model()
        self.optimizers[self.model_types[0]] = getattr(optim, params['optimizer'])(
            self.models[self.model_types[0]].parameters(),
            lr=self.config.lr_opt["init"],
            weight_decay=self.config.l2_penalty)

        # Reset Training Parameters
        self.config.lr_opt = self.init_lr_config
        self.lr = self.config.lr_opt["init"]
        self.complete_epochs = 0
        self.best_metric, self.best_epoch = float('-inf'), 0

        self.train(tuning_trial=trial, params=params)
        self.load_model()
        score = self.test()

        self.tuning_log += [dict(params=params, score=score)]
        with open(f'{self.tuning_log_path}.txt', 'a+') as f:
            for trial in self.tuning_log:
                line = str(trial['params']) + '  ' + str(trial['score'])
                f.write(line)
                f.write('\n')

        return score['dice']

    def tuning(self):
        self.study = optuna.create_study(direction="maximize",
                                         sampler=optuna.samplers.TPESampler(),
                                         pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))

        # Add Previously tested conditions
        prev_trials = joblib.load('tuning_log_new_dice/ResUNetMultiConnection_2022-04-17 12-30-30.pkl').trials
        self.study.add_trials(prev_trials)

        try:
            self.study.optimize(self.objective, n_trials=10)
        except:
            print('Error during tuning! Emergency Exit...')
            with open(f'{self.tuning_log_path}.txt', 'a+') as f:
                f.write("Error during tuning! Emergency Exit...\n")

        # logfile_name = f"{self.config.model_type}_{str(datetime.now())[:-7].replace(':', '-')}"

        # Save study object file
        joblib.dump(self.study, f"{self.tuning_log_path}.pkl")

        # Save .txt log file
        lines = []
        lines += [str(datetime.now()) + '\n']
        lines += [f'Tuning {self.config.model_type} ...']
        lines += ['Best Parameters :']
        lines += [str(self.study.best_params)]
        lines += [str(self.study.best_value)]

        with open(f'{self.tuning_log_path}.txt', 'a+') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

        # with open(f'tuning_log/{logfile_name}.txt', 'a+') as f:
        #     f.write("\nTrials .... \n")
        #     for trial in self.tuning_log:
        #         line = str(trial['params']) + '  ' + str(trial['score'])
        #         f.write(line)
        #         f.write('\n')
