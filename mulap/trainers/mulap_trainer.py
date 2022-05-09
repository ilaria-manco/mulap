import os
import time
import numpy as np

import torch
from torch import nn
from torch.optim import Adam, Adadelta
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from mulap.datasets.audiocaption import AudioCaptionDataset
from mulap.trainers.base_trainer import BaseTrainer
from mulap.models.mulbert import MuLBertForPretraining
from mulap.models.config import MultimodalBertConfig


class MuLBertTrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.bert_config = self.config.model_config.bert

        self.load()

        self.scaler = torch.cuda.amp.GradScaler()

    def load_dataset(self):
        self.logger.write("Loading dataset")
        dataset_name = self.config.dataset_config.dataset_name
        atm_task = self.bert_config.multimodal_objective == "atm"

        if dataset_name == "audiocaption":
            train_dataset = AudioCaptionDataset(
                self.config.dataset_config, atm_task=atm_task)
            val_dataset = AudioCaptionDataset(
                self.config.dataset_config, dataset_type="val", atm_task=atm_task
            )
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

    def build_model(self):
        self.logger.write("Building model")

        config = MultimodalBertConfig(self.bert_config)
        if self.config.model_config.pretrained_bert is not None:
            self.logger.write(
                "Initializing BERT weights with pretrained {}".format(
                    self.config.model_config.pretrained_bert
                )
            )
            self.model = MuLBertForPretraining.from_pretrained(
                pretrained_model_name_or_path=self.config.model_config.pretrained_bert,
                config=config,
                audio_config=self.config.model_config.audio,
            )
        else:
            self.logger.write(
                "No pretrained BERT initialisation. The text branch will be trained from scratch."
            )
            self.model = MuLBertForPretraining(
                config, self.config.model_config.audio)

        if self.model.audio_backbone.pretrained_version:
            state_dict = torch.load(
                self.config.model_config.audio.feature_extractor_path
            )
            self.model.audio_backbone.feature_extractor.load_state_dict(
                state_dict, strict=False
            )
            if not self.config.model_config.audio.finetune:
                for param in self.model.audio_backbone.feature_extractor.parameters():
                    param.requires_grad = False

        if self.config.model_config.freeze_text_branch:
            self.logger.write("Freezing text branch")
            for name, param in self.model.named_parameters():
                text_branch = [
                    "bert.embeddings",
                    "bert.encoder.layer",
                    "bert.t_pooler",
                    "cls.text_predictions",
                ]
                if any(s in name for s in text_branch):
                    param.requires_grad = False

        self.model.to(self.device)

    def build_optimizer(self):
        self.logger.write("Building optimizer")
        optimizer_name = self.config.training.optimizer
        if optimizer_name == "adam":
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.config.training.lr)
        elif optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.training.lr,
                eps=self.config.training.adam_epsilon,
                betas=self.config.training.adam_betas,
            )
            num_train_optimization_steps = (
                int(
                    self.train_loader.dataset.__len__()
                    / self.config.training.batch_size
                    / self.config.training.grad_acc_steps
                )
                * self.config.training.epochs
            )
            warmup_steps = (
                self.config.training.warmup_proportion * num_train_optimization_steps
            )
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps,
            )
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.config.training.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        elif optimizer_name == "adadelta":
            self.optimizer = Adadelta(
                self.model.parameters(), lr=self.config.training.lr
            )
        else:
            raise ValueError(
                "{} optimizer is not supported.".format(optimizer_name))

    def train(self):
        best_val_loss = np.inf
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

            train_loss = self.train_epoch(self.train_loader, is_training=True)
            val_loss = self.train_epoch_val(self.val_loader)

            epoch_time = time.time() - epoch_start_time
            self.logger.update_training_log(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                self.scheduler.get_last_lr()[0],
            )

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            # save checkpoint in appropriate path (new or best)
            self.logger.save_checkpoint(state=checkpoint, is_best=is_best)

    def load_ckp(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"]

    def train_epoch(self, data_loader, is_training):
        running_loss = 0.0
        n_batches = 0

        if is_training:
            self.model.train()
            if self.model.audio_backbone.pretrained_version is not None:
                for module in self.model.audio_backbone.feature_extractor.modules():
                    if isinstance(module, nn.BatchNorm2d) or isinstance(
                        module, nn.BatchNorm1d
                    ):
                        module.eval()
        else:
            self.model.eval()

        for i, batch in enumerate(data_loader):
            batch = tuple(t.to(device=self.device) for t in batch)
            (
                audio_id,
                input_audio,
                audio_attention_mask,
                text_input_ids,
                text_input_type_ids,
                text_attention_mask,
                mlm_labels,
                atm_label,
            ) = batch

            if self.config.model_config.atm_loss_weight != 0:
                # Ignore labels (setting them to -1) for mismatched caption-audio pairs
                mlm_labels = mlm_labels * (atm_label == 0).long().unsqueeze(1)
                mlm_labels[mlm_labels == 0] = -1

            # Cast operations to mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.training.amp):
                masked_loss_t, atm_loss, masked_loss_a = self.model(
                    input_audio=input_audio,
                    text_input_ids=text_input_ids,
                    text_input_type_ids=text_input_type_ids,
                    text_attention_mask=text_attention_mask,
                    mlm_labels=mlm_labels,
                    atm_label=atm_label,
                )

            masked_loss_t = self.config.model_config.mlm_loss_weight * masked_loss_t
            masked_loss_a = self.config.model_config.audio_loss_weight * masked_loss_a
            atm_loss = self.config.model_config.atm_loss_weight * atm_loss
            loss = masked_loss_t + atm_loss + masked_loss_a

            # If training, backprop and optimize
            if is_training:
                if self.config.training.amp:
                    if self.config.training.grad_acc_steps > 1:
                        loss = loss / self.config.training.grad_acc_steps

                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    self.scaler.scale(loss).backward()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()

        n_batches += 1
        return running_loss / n_batches

    def train_epoch_val(self, data_loader):
        with torch.no_grad():
            loss = self.train_epoch(data_loader, is_training=False)
        return loss
