import os
import time
import shutil
import numpy as np
import torch


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 valid_metric_name="-loss",
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger
        self.is_decreased_valid_metric = valid_metric_name[0] == "-"
        self.valid_metric_name = valid_metric_name[1:]
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler

        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaulation " + "-" * 33

    def train_epoch(self):
        """
        train_epoch
        """
        self.epoch += 1
        num_batches = len(self.train_iter)
        self.logger.info(self.train_start_message)

        for batch_id, inputs in enumerate(self.train_iter, 1):
            self.model.train()
            start_time = time.time()
            # training iteration
            metrics_dict = self.model.iterate(inputs, optimizer=self.optimizer,
                                              grad_clip=self.grad_clip,
                                              is_training=True)
            elapsed = time.time() - start_time
            self.batch_num += 1

            if batch_id % self.log_steps == 0:
                message_prefix = "[Train][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = "Loss:{:.3f} Loss_g:{:.3f} Loss_v:{:.3f} Loss_l:{:.3f}".format(
                    metrics_dict['loss'], metrics_dict['loss_g'], metrics_dict['loss_v'], metrics_dict['loss_l'])
                message_posfix = "TIME:{:.2f}".format(elapsed)
                self.logger.info("   ".join([message_prefix, metrics_message, message_posfix]))

            if batch_id % self.valid_steps == 0:
                self.logger.info(self.valid_start_message)
                valid_mm_dict = self.evaluate(self.model, self.valid_iter)

                message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = "Loss:{:.3f} Loss_g:{:.3f} Loss_v:{:.3f} Loss_l:{:.3f}".format(
                    valid_mm_dict['loss'], valid_mm_dict['loss_g'], valid_mm_dict['loss_v'], valid_mm_dict['loss_l'])
                self.logger.info("   ".join([message_prefix, metrics_message]))

                cur_valid_metric = valid_mm_dict['loss']
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric

                self.save(is_best=is_best)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(cur_valid_metric)
                self.logger.info("-" * 85 + "\n")

        self.save()
        self.logger.info('')

    def evaluate(self, model, data_iter):
        model.eval()
        loss = []
        loss_g = []
        loss_v = []
        loss_l = []
        with torch.no_grad():
            for inputs in data_iter:
                metrics = model.iterate(inputs, is_training=False)
                loss.append(metrics['loss'].item())
                loss_g.append(metrics['loss_g'].item())
                loss_v.append(metrics['loss_v'].item())
                loss_l.append(metrics['loss_l'].item())
        loss_dict = {"loss": np.mean(loss),
                     "loss_g": np.mean(loss_g),
                     "loss_v": np.mean(loss_v),
                     "loss_l": np.mean(loss_l)
                     }
        return loss_dict

    def train(self):
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()

    def save(self, is_best=False):
        """
        save
        """
        model_file = os.path.join(self.save_dir, "epoch_{}.model".format(self.epoch))
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file = os.path.join(self.save_dir, "epoch_{}.train".format(self.epoch))
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}:{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, save_dir, file_prefix):
        """
        load
        """
        model_file = "{}/{}.model".format(save_dir, file_prefix)
        train_file = "{}/{}.train".format(save_dir, file_prefix)

        model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))
