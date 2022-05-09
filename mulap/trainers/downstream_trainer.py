import os
import time
import numpy as np
import copy
from sklearn import metrics

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Subset

from mulap.datasets.nsynth import Nsynth
from mulap.datasets.emomusic import Emomusic
from mulap.datasets.fma_small import FMASmall
from mulap.trainers.base_trainer import BaseTrainer
from mulap.datasets.tagging import MTTDataset, JamendoDataset
from mulap.evaluation.clf_evaluation import LinearClassifier


class Downstream(BaseTrainer):
    def __init__(self, pretrain_config, downstream_config, logger):
        super().__init__(config=downstream_config, logger=logger)
        self.pretrain_config = pretrain_config

        self.load()
        self.build_loss()

    def load_dataset(self):
        dataset_name = self.config.dataset_name
        self.logger.write("Loading {} dataset".format(dataset_name))
        data_root = os.path.join(
            self.pretrain_config.env.data_root, "datasets", dataset_name
        )

        if dataset_name == "mtt":
            train_dataset = MTTDataset(data_root)
            val_dataset = MTTDataset(data_root, subset="validation")
            self.config.num_classes = train_dataset.num_classes()
        elif dataset_name == "jamendo":
            train_dataset = JamendoDataset(data_root, self.config.category)
            val_dataset = JamendoDataset(
                data_root, self.config.category, subset="validation"
            )
            self.config.num_classes = train_dataset.num_classes()

            if self.config.samples_per_class is not None:
                genres = []
                train_idx = []
                for index, data in enumerate(train_dataset):
                    _, label = data
                    if genres.count(label) < self.config.samples_per_class:
                        train_idx.append(index)
                        genres.append(label)
                train_dataset = Subset(train_dataset, train_idx)

        elif dataset_name == "nsynth":
            train_dataset = Nsynth(data_root, subset="training")
            val_dataset = Nsynth(data_root, subset="validation")
            self.config.num_classes = train_dataset.num_classes()
        elif dataset_name == "emomusic":
            train_dataset = Emomusic(data_root, subset="training")
            val_dataset = Emomusic(data_root, subset="validation")
            self.config.num_classes = train_dataset.num_classes()
        elif dataset_name == "fma":
            train_dataset = FMASmall(data_root, subset="training")
            val_dataset = FMASmall(data_root, subset="validation")
            self.config.num_classes = train_dataset.num_classes()
        else:
            raise ValueError(
                "{} dataset is not supported.".format(dataset_name))

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
        )

        self.logger.write(
            "Number of training samples: {}".format(train_dataset.__len__())
        )

    def load_mulbert_layers(self):
        pretrained_mulbert_path = os.path.join(
            self.config.pretrain_experiment_dir, "best_model.pth.tar"
        )
        mulbert_state_dict = torch.load(pretrained_mulbert_path)["state_dict"]
        mulbert_state_dict_v2 = copy.deepcopy(mulbert_state_dict)
        for key in mulbert_state_dict:
            if "audio_embeddings" in key:
                new_key = key.replace("mulbert.", "")
                mulbert_state_dict_v2[new_key] = mulbert_state_dict_v2.pop(key)

        self.model.load_state_dict(mulbert_state_dict_v2, strict=False)

    def build_model(self):
        self.logger.write("Building model")
        self.model = LinearClassifier(self.pretrain_config, self.config)
        self.model.to(self.device)

        if self.config.backbone_init == "no_backbone":
            pass
        else:
            if self.config.backbone_init == "mulap":
                self.logger.write(
                    "Initiating audio backbone weights with mulap initialisation"
                )
                self.load_mulbert_layers()
            elif "supervised" in self.config.backbone_init:
                self.logger.write(
                    "Initiating audio backbone weights with supervised initialisation"
                )
                backbone_path = (
                    self.pretrain_config.model_config.audio.feature_extractor_path
                )
                backbone_path = backbone_path.replace(
                    "None", self.config.backbone_init.split("_")[1]
                )
                try:
                    backbone_state_dict = torch.load(backbone_path)
                except:
                    raise ValueError(
                        "Backbone trained on {} in a supervised way is not supported".format(
                            self.config.backbone_init.split("_")[1]
                        )
                    )
                self.model.audio_backbone.feature_extractor.load_state_dict(
                    backbone_state_dict, strict=False
                )
            elif self.config.backbone_init == "random":
                self.logger.write(
                    "Initiating audio backbone weights with random initialisation"
                )
                pass
            else:
                raise ValueError(
                    "{} backbone initialisation is not supported.".format(
                        self.config.backbone_init
                    )
                )
        if self.config.freeze_backbone:
            self.logger.write("Freezing audio backbone layers")
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            if self.config.bert_audio_embed:
                self.model.audio_embeddings.eval()

    def build_loss(self):
        if self.config.task == "multilabel_clf":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.config.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.task == "regression":
            self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(self.device)

    def build_optimizer(self):
        optimizer_name = self.config.training.optimizer
        if optimizer_name == "adam":
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config.training.lr,
                weight_decay=0
                if self.config.training.weight_decay is None
                else self.config.training.weight_decay,
            )
        else:
            raise ValueError(
                "{} optimizer is not supported.".format(optimizer_name))

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min" if self.config.training.early_stop.minimise else "max",
            factor=0.5,
        )

    def train(self):
        if self.config.training.early_stop.minimize:
            best_early_stop_metric = np.inf
        else:
            best_early_stop_metric = 0
        patience = 0

        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write(
                "Resumed training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.load_ckp(self.logger.checkpoint_path)
        else:
            self.logger.write(
                "Started training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()

            train_loss = self.training_loop(self.train_loader)
            val_loss, task_metric = self.validation_loop(self.val_loader)

            epoch_time = time.time() - epoch_start_time
            self.logger.update_training_log(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                self.optimizer.param_groups[0]["lr"],
                task_metric,
            )

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            if self.config.training.early_stop.criteria == "loss":
                early_stop_metric = val_loss
            elif self.config.training.early_stop.criteria == "task_metric":
                early_stop_metric = task_metric
            if self.config.training.early_stop.minimize:
                is_best = early_stop_metric < best_early_stop_metric
            else:
                is_best = early_stop_metric > best_early_stop_metric
            if is_best:
                best_early_stop_metric = early_stop_metric
                patience = 0
            else:
                patience += 1

            self.scheduler.step(early_stop_metric)

            # save checkpoint in appropriate path (new or best)
            self.logger.save_checkpoint(state=checkpoint, is_best=is_best)

            if self.config.training.early_stop.enabled:
                if patience >= self.config.training.early_stop.patience:
                    self.logger.write("Early stopping")
                    break

    def load_ckp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"]

    def training_step(self, batch, batch_idx):
        audio, labels = batch
        audio = audio.to(device=self.device)
        labels = labels.to(device=self.device)

        logits = self.model(audio)

        if len(logits.size()) == 3:
            logits = torch.mean(logits, dim=1)
            logits = logits.squeeze(1)

        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            labels = labels.type_as(logits)
            out = torch.sigmoid(logits)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            out = torch.argmax(torch.log_softmax(logits, dim=1), dim=1)
        elif isinstance(self.criterion, nn.MSELoss):
            out = logits
        loss = self.criterion(logits, labels)

        return loss, out, labels

    def training_loop(self, data_loader):
        self.model.train()

        running_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(data_loader):
            loss, _, _ = self.training_step(batch, i)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        return running_loss / n_batches

    def validation_loop(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            running_loss = 0.0
            n_batches = 0

            if self.config.task == "multilabel_clf":
                ground_truth = np.zeros(
                    (self.val_loader.dataset.__len__(), self.config.num_classes)
                )
                predictions = np.zeros(
                    (self.val_loader.dataset.__len__(), self.config.num_classes)
                )
            elif self.config.task == "classification":
                ground_truth = np.zeros((self.val_loader.dataset.__len__(),))
                predictions = np.zeros((self.val_loader.dataset.__len__(),))
            elif self.config.task == "regression":
                ground_truth = np.zeros((self.val_loader.dataset.__len__(), 2))
                predictions = np.zeros((self.val_loader.dataset.__len__(), 2))

            for i, batch in enumerate(data_loader):
                loss, out, labels = self.training_step(batch, i)

                batch_size = len(batch[0])

                ground_truth[
                    i * batch_size: (i + 1) * batch_size
                ] = labels.cpu().numpy()
                predictions[i * batch_size: (i + 1)
                            * batch_size] = out.cpu().numpy()

                running_loss += loss.item()
                n_batches += 1

            if self.config.task == "multilabel_clf":
                task_metric = metrics.roc_auc_score(
                    ground_truth, predictions, average="macro"
                )
            elif self.config.task == "classification":
                task_metric = metrics.accuracy_score(ground_truth, predictions)
            elif self.config.task == "regression":
                task_metric = metrics.r2_score(ground_truth, predictions)

        return running_loss / n_batches, task_metric
