import numpy as np
import torch
import torch.nn as nn
from base import BaseTrainer
from utils import inf_loop, MetricTracker


from torch import sigmoid


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, batch in enumerate(self.data_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            targets = batch["correct"]

            if self.config["arch"]["type"] == "LastQueryModel":
                targets = targets[:, -1]
                loss = self.criterion(output=outputs, target=targets.float())
                loss = torch.mean(loss)

                outputs = sigmoid(outputs)

            else:
                loss = self.criterion(output=outputs, target=targets.float())
                loss = loss[:, -1]
                loss = torch.mean(loss)

                outputs = sigmoid(outputs[:, -1])
                targets = targets[:, -1]

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config["arch"]["args"]["clip_grad"])

            if self.config["lr_scheduler"]["type"] == "linear_warmup":
                self.lr_scheduler.step()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
                )

            if batch_idx % self.log_step == 0:
                # self.logger.info("Training steps: %s Loss: %.4f", step, loss.item())
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), loss.item())
                )
                # self.writer.add_image('input', make_grid(batch.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(log["val_auc_score"])  # val auc

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                targets = batch["correct"]

                if self.config["arch"]["type"] == "LastQueryModel":
                    outputs = sigmoid(outputs)
                else:
                    outputs = sigmoid(outputs[:, -1])

                targets = targets[:, -1]

                loss = self.criterion(outputs, targets.float())
                loss = torch.mean(loss)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
                    )
                # self.writer.add_image('input', make_grid(batch.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


####################################################################################################


import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode="max", verbose=True)
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    return scheduler
