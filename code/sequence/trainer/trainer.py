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

        total_outputs = []
        total_targets = []
        for batch_idx, batch in enumerate(self.data_loader):
            batch = self.process_batch(batch)
            print(batch[0].shape)

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            targets = batch[0] # data['correct']
            index = batch[-1]

            loss = self.criterion(outputs, targets)
            loss = torch.gather(loss, 1, index)
            loss = torch.mean(loss)

########

            # if self.config["arch"]["type"] == "LastQueryModel":
            #     targets = targets[:, -1]
            #     loss = self.criterion(output=outputs, target=targets.float())
            #     loss = torch.mean(loss)

            #     # outputs = sigmoid(outputs)

            # else:
            #     loss = self.criterion(output=outputs, target=targets.float())
            #     loss = loss[:, -1]
            #     loss = torch.mean(loss)

            #     # outputs = sigmoid(outputs[:, -1])
            #     targets = targets[:, -1]

########

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config["arch"]["args"]["clip_grad"])

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.config["lr_scheduler"]["type"] == "linear_warmup":
                self.lr_scheduler.step()

            # preidctions
            outputs = outputs.gather(1, index).view(-1)
            targets = targets.gather(1, index).view(-1)

            total_outputs.append(outputs.cpu().detach().numpy())
            total_targets.append(targets.cpu().detach().numpy())

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())

            if batch_idx % self.log_step == 0:
                # self.logger.info("Training steps: %s Loss: %.4f", step, loss.item())
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), loss.item())
                )
                # self.writer.add_image('input', make_grid(batch.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        total_outputs = np.concatenate(total_outputs)
        total_targets = np.concatenate(total_targets)

        for met in self.metric_ftns:
            self.train_metrics.update(
                met.__name__, met(total_outputs, total_targets)
            )

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

        total_outputs = []
        total_targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                batch = self.process_batch(batch)

                outputs = self.model(batch)
                targets = batch[0]  # data['correct']
                index = batch[-1]

                loss = self.criterion(outputs, targets)
                loss = torch.gather(loss, 1, index)
                loss = torch.mean(loss)

########
                # if self.config["arch"]["type"] == "LastQueryModel":
                #     outputs = sigmoid(outputs)
                # else:
                #     outputs = sigmoid(outputs[:, -1])

                # targets = targets[:, -1]

                # loss = self.criterion(outputs, targets.float())
########

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())

                # predictions
                outputs = outputs.gather(1, index).view(-1)
                targets = targets.gather(1, index).view(-1)

                total_outputs.append(outputs.cpu().detach().numpy())
                total_targets.append(targets.cpu().detach().numpy())
                
        total_outputs = np.concatenate(total_outputs)
        total_targets = np.concatenate(total_targets)
            
        for met in self.metric_ftns:
            self.valid_metrics.update(
                met.__name__, met(total_outputs, total_targets)
            )

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
    
    # 배치 전처리
    def process_batch(self, batch):

        correct, question, test, tag, elapsed_question, elapsed_test, mask = batch

        # change to float
        mask = mask.type(torch.FloatTensor)
        correct = correct.type(torch.FloatTensor)

        #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
        #    saint의 경우 decoder에 들어가는 input이다
        interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1, dims=1)
        interaction[:, 0] = 0 # set padding index to the first sequence
        interaction = (interaction * mask).to(torch.int32)

        #  question_id, test_id, tag
        question = ((question + 1) * mask).to(torch.int32)
        test = ((test + 1) * mask).to(torch.int32)
        tag = ((tag + 1) * mask).to(torch.int32)
        elapsed_question = ((elapsed_question + 1) * mask).to(torch.int32)
        elapsed_test = ((elapsed_test + 1) * mask).to(torch.int32)

        # gather index
        # 마지막 sequence만 사용하기 위한 index
        gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
        gather_index = gather_index.view(-1, 1) - 1

        # device memory로 이동
        correct = correct.to(self.device)
        question = question.to(self.device)
        test = test.to(self.device)
        tag = tag.to(self.device)
        elapsed_question = elapsed_question.to(self.device)
        elapsed_test = elapsed_test.to(self.device)
        
        mask = mask.to(self.device)
        interaction = interaction.to(self.device)
        gather_index = gather_index.to(self.device)

        return (correct, question, test, tag,  
                elapsed_question, elapsed_test,
                mask, interaction, gather_index)



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
