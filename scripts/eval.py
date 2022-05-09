import os
import argparse
from omegaconf import OmegaConf

from mulap.utils.utils import get_root_dir
from mulap.evaluation.clf_evaluation import ClfEvaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a music captioning model")

    parser.add_argument("pretrain_id", type=str)
    parser.add_argument("downstream_id", type=str)
    parser.add_argument("--save_output", type=bool, default=True)
    parser.add_argument("--device_num", type=str, default="0")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    pretrain_id = params.pretrain_id
    downstream_id = params.downstream_id
    base_dir = get_root_dir()

    pretrain_config = OmegaConf.load(os.path.join(
        get_root_dir(), "save/experiments/{}/config.yaml".format(pretrain_id)))
    downstream_config = OmegaConf.load(os.path.join(
        get_root_dir(), "save/experiments/{}/downstream/{}/config.yaml".format(pretrain_id, downstream_id)))

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num
    evaluation = ClfEvaluation(
        pretrain_config, downstream_config, params.save_output)
    evaluation.get_metrics()
