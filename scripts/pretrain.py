import os
import argparse
from omegaconf import OmegaConf

from mulap.utils.logger import Logger
from mulap.datasets.audiocaption import AudioCaptionDataset
from mulap.models.mulbert import MuLBertForPretraining
from mulap.trainers.mulap_trainer import MuLBertTrainer
from mulap.utils.utils import load_conf, merge_conf, get_root_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run MuLaP pre-training")

    parser.add_argument(
        "--experiment_id",
        type=str,
        help="experiment id under which checkpoint was saved",
        default=None,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to base config file",
        default=os.path.join(get_root_dir(), "configs", "default.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of the dataset",
        default="audiocaption",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="version of pretrained model",
        default=None,
    )
    parser.add_argument(
        "--finetune",
        type=str,
        help="whether to finetune audio feature extractor",
        default=None,
    )
    parser.add_argument(
        "--track_metrics",
        type=str,
        help="whether to track metrics while training",
        default=None,
    )
    parser.add_argument(
        "--device_num",
        type=str,
        default="0",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    if params.experiment_id is None:
        # 1. Load config (base + dataset + model)
        base_conf = load_conf(params.config_path)

        if params.dataset == "audiocaption":
            dataset_conf_path = os.path.join(
                base_conf.env.base_dir, AudioCaptionDataset.config_path())
        else:
            print("{} dataset not supported".format(params.dataset))

        model_conf_path = os.path.join(
            base_conf.env.base_dir, MuLBertForPretraining.config_path()
        )
        config = merge_conf(params.config_path,
                            dataset_conf_path, model_conf_path)

        # Update config values with command line args if input
        if params.pretrained_model is not None:
            OmegaConf.update(
                config, "model_config.pretrained_version", params.pretrained_model
            )
    else:
        config = OmegaConf.load(
            "./save/experiments/{}/config.yaml".format(params.experiment_id)
        )

    logger = Logger(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num

    trainer = MuLBertTrainer(config, logger)
    trainer.train()
