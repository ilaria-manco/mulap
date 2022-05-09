import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader

from mulap.datasets.nsynth import Nsynth
from mulap.datasets.fma_small import FMASmall
from mulap.datasets.emomusic import Emomusic
from mulap.datasets.tagging import MTTDataset, JamendoDataset
from mulap.modules.audio_encoders import CNNEncoder
from mulap.modules.bert_audio_layers import BertAudioEmbeddings


class LinearClassifier(nn.Module):
    def __init__(self, pretrain_config, downstream_config):
        super().__init__()
        self.pretrain_config = pretrain_config
        self.downstream_config = downstream_config
        num_classes = self.downstream_config.num_classes

        if self.downstream_config.backbone_init == "no_backbone":
            self.feature_dim = self.downstream_config.feature_dim
            self.audio_backbone = nn.Identity()
        else:
            self.audio_backbone = CNNEncoder(
                self.pretrain_config.model_config.audio)
            self.feature_dim = self.audio_backbone.audio_feature_dim
        if downstream_config.bert_audio_embed:
            print("adding bert audio embedding layer")
            self.audio_embeddings = BertAudioEmbeddings(
                self.pretrain_config.model_config.bert.audio
            )
            self.feature_dim = self.pretrain_config.model_config.bert.audio.hidden_size

        self.dropout = nn.Dropout(p=self.downstream_config.dropout_p)

        if self.downstream_config.classifier == "logistic_regression":
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        elif self.downstream_config.classifier == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.downstream_config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.downstream_config.hidden_dim, num_classes),
                self.dropout,
            )

    def forward(self, x):
        x = self.audio_backbone(x)
        x = self.dropout(x)
        if self.downstream_config.bert_audio_embed:
            x = self.audio_embeddings(x)
        logits = self.classifier(x)
        return logits


class ClfEvaluation:
    def __init__(self, pretrain_config, downstream_config, save_output):
        self.pretrain_config = pretrain_config
        self.downstream_config = downstream_config
        self.device = torch.device(self.downstream_config.training.device)
        self.downstream_experiment_id = self.downstream_config.downstream_experiment_id
        self.path_to_model = os.path.join(self.downstream_config.downstream_dir,
                                          self.downstream_experiment_id,
                                          "best_model.pth.tar")

        self.task = self.downstream_config.task

        self.output_path = os.path.join(
            self.downstream_config.downstream_dir, self.downstream_experiment_id, "predictions.npz")
        self.save_output = save_output

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        dataset_name = self.downstream_config.dataset_name
        data_root = os.path.join(
            self.pretrain_config.env.data_root, "datasets", dataset_name)

        if dataset_name == "mtt":
            test_dataset = MTTDataset(data_root, subset="testing")
        elif dataset_name == "nsynth":
            test_dataset = Nsynth(data_root, subset='testing')
        elif dataset_name == "emomusic":
            test_dataset = Emomusic(data_root, subset='testing')
        elif dataset_name == "jamendo":
            test_dataset = JamendoDataset(
                data_root, self.downstream_config.category, subset='testing')
        elif dataset_name == "fma":
            test_dataset = FMASmall(data_root, subset='testing')
        else:
            raise ValueError(
                "{} dataset is not supported.".format(dataset_name))
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    def build_model(self):
        self.model = LinearClassifier(
            self.pretrain_config, self.downstream_config)
        self.model.to(self.device)

        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def obtain_predictions(self):
        if not os.path.exists(self.output_path):
            ground_truth = []
            predicted = []
            with torch.no_grad():
                for i, batch in enumerate(tqdm(self.test_loader)):
                    audio, label = batch
                    audio = audio.to(device=self.device)
                    label = label.to(device=self.device)

                    logits = self.model(audio)

                    if self.task == "multilabel_clf":
                        output = torch.sigmoid(logits)
                        # average across chunks in a track
                        track_prediction = output.mean(dim=1)
                        predicted.append(
                            track_prediction.squeeze().cpu().numpy())
                    elif self.task == "classification":
                        if len(logits.size()) == 3:
                            output = torch.log_softmax(logits, dim=2)
                            # average across chunks in a track
                            track_prediction = torch.argmax(
                                output.mean(dim=1).squeeze())
                        else:
                            track_prediction = torch.argmax(
                                torch.log_softmax(logits, dim=1))
                        predicted.append(track_prediction.cpu().numpy())
                    elif self.task == "regression":
                        logits = logits.mean(dim=1).squeeze()
                        predicted.append(logits.cpu().numpy())
                    ground_truth.append(label.squeeze().cpu().numpy())

            predicted = np.array(predicted)
            ground_truth = np.array(ground_truth)
            if self.save_output:
                np.savez(self.output_path, predicted, ground_truth)
        else:
            saved_output = np.load(self.output_path)
            predicted = saved_output['arr_0']
            ground_truth = saved_output['arr_1']
        return predicted, ground_truth

    def get_binary_decisions(self, tags, ground_truth, predicted):
        """https://github.com/MTG/mtg-jamendo-dataset/blob/31507d6e9a64da11471bb12168372db4f98d7783/scripts/mediaeval/calculate_decisions.py#L8"""
        thresholds = {}
        for i, tag in enumerate(tags):
            precision, recall, threshold = metrics.precision_recall_curve(
                ground_truth[:, i], predicted[:, i])
            f_score = np.nan_to_num(
                (2 * precision * recall) / (precision + recall))
            thresholds[tag] = threshold[np.argmax(f_score)]

        for tag, threshold in thresholds.items():
            print('{}\t{:6f}'.format(tag, threshold))

        decisions = predicted > np.array(list(thresholds.values()))

        return thresholds, decisions

    def get_metrics(self, show_results=True, output_file=None):
        predicted, ground_truth = self.obtain_predictions()
        results = {}

        if self.task == "multilabel_clf":
            for average in ['macro', 'micro']:
                results['PR-AUC-' + average] = metrics.average_precision_score(
                    ground_truth, predicted, average=average)
                results['ROC-AUC-' + average] = metrics.roc_auc_score(
                    ground_truth, predicted, average=average)
                tags = range(self.test_loader.dataset.num_classes())
                _, y_pred = self.get_binary_decisions(
                    tags, ground_truth, predicted)
                results['F1-' +
                        average] = metrics.f1_score(ground_truth, y_pred, average=average)
            results['accuracy'] = metrics.accuracy_score(ground_truth, y_pred)
        elif self.task == "classification":
            results['accuracy'] = metrics.accuracy_score(
                ground_truth, predicted)
            results['f1_score'] = metrics.f1_score(
                ground_truth, predicted, average='macro')
        elif self.task == "regression":
            results['r2'] = metrics.r2_score(ground_truth, predicted)
            results["arousal_r2"] = metrics.r2_score(
                ground_truth[:, 0], predicted[:, 0])
            results["valence_r2"] = metrics.r2_score(
                ground_truth[:, 1], predicted[:, 1])

        if show_results:
            for metric, value in results.items():
                print('{:20}\t{:6f}'.format(metric, value))

        if output_file is not None:
            df = pd.DataFrame(results.values(), results.keys())
            df.to_csv(output_file, sep='\t', header=None, float_format='%.6f')

        return results
