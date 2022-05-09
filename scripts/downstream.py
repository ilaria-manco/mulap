import os
import argparse
from omegaconf import OmegaConf

from mulap.utils.logger import Logger
from mulap.trainers.downstream_trainer import Downstream
from mulap.utils.utils import get_root_dir, load_conf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a music captioning model")

    parser.add_argument(
        "pretrained_model_id",
        type=str,
        help="pre-trained model ID",
    )
    parser.add_argument(
        "downstream_task", type=str, help="name of the downstream evaluation"
    )
    parser.add_argument(
        "--classifier", type=str, help="type of classifier to use", default=None
    )
    parser.add_argument(
        "--backbone_init", type=str, help="backbone initialisation", default=None
    )
    parser.add_argument(
        "--freeze_backbone",
        type=bool,
        help="whether to freeze backbone weights",
        default=None,
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        help="experiment id under which checkpoint was saved",
        default=None,
    )
    parser.add_argument("--device_num", type=str, default="0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    pretrained_model_id = params.pretrained_model_id
    base_dir = get_root_dir()

    # load config for pretrained model (this has all the settings on how it was trained)
    pretrain_config = OmegaConf.load(
        os.path.join(
            base_dir, "save/experiments/{}/config.yaml".format(
                pretrained_model_id)
        )
    )
    pretrain_experiment_dir = os.path.join(
        base_dir, "save/experiments/", pretrained_model_id
    )

    if params.experiment_id is None:
        # load and save config for downstream training
        downstream_config = load_conf(
            os.path.join(
                base_dir, "configs/downstream/{}.yaml".format(
                    params.downstream_task)
            )
        )
        OmegaConf.update(
            downstream_config, "pretrain_experiment_dir", pretrain_experiment_dir
        )

        if params.classifier is not None:
            OmegaConf.update(downstream_config,
                             "classifier", params.classifier)
        if params.backbone_init is not None:
            OmegaConf.update(downstream_config,
                             "backbone_init", params.backbone_init)
        if params.freeze_backbone is not None:
            freeze_backbone = params.freeze_backbone == "true"
            OmegaConf.update(downstream_config,
                             "freeze_backbone", freeze_backbone)
    else:
        downstream_config = OmegaConf.load(
            os.path.join(
                get_root_dir(),
                "save/experiments/{}/downstream/{}/config.yaml".format(
                    pretrained_model_id, params.experiment_id
                ),
            )
        )

    logger = Logger(downstream_config)

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num
    trainer = Downstream(pretrain_config, downstream_config, logger)
    trainer.train()
