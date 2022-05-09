import torch
from torch import nn


class RegressionLoss(nn.Module):
    def __init__(self, loss_name):
        super().__init__()
        if loss_name == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        elif loss_name == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="none")

    def forward(self, prediction_scores_a, audio_target, audio_labels):
        audio_loss = self.criterion(prediction_scores_a, audio_target)
        masked_audio_loss = torch.sum(
            audio_loss * (audio_labels == 1).unsqueeze(2).float()) / max(torch.sum((audio_labels == 1).unsqueeze(2).expand_as(audio_loss)), 1)

        return masked_audio_loss


class KLMaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="none")

    def forward(self, prediction_scores_a, audio_target, audio_labels):
        audio_loss = self.criterion(F.log_softmax(
            prediction_scores_a, dim=2), audio_target)
        masked_audio_loss = torch.sum(
            audio_loss * (audio_labels == 1).unsqueeze(2).float()) / max(torch.sum((audio_labels == 1)), 0)

        return masked_audio_loss
