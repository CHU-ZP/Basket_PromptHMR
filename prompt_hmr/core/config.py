import os
import shutil
import argparse
from yacs.config import CfgNode as CN
from data_config import *


# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.DEVICE = 'cuda'
cfg.LOGDIR = ''
cfg.NUM_WORKERS = 4
cfg.SEED_VALUE = -1
cfg.IMG_RES = 224
cfg.PIN_MEMORY = False

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.RESUME = ''
cfg.TRAIN.PRETRAINED = ''
cfg.TRAIN.IS_FINETUNE = False
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 100
cfg.TRAIN.WARMUP_STEPS = 1
cfg.TRAIN.LR_SCHEDULE = []
cfg.TRAIN.LR_DECAY = 1.0
cfg.TRAIN.MAX_STEP = 100000
cfg.TRAIN.CLIP_GRADIENT = 1.0
cfg.TRAIN.WD = 0.01
cfg.TRAIN.OPT = 'AdamW'
cfg.TRAIN.GRAD_ACC = 1
cfg.TRAIN.FINETUNE = None

cfg.DATASET = CN()
cfg.DATASET.TEST = 'emdb_1'
cfg.DATASET.SEQ_LEN = 16
cfg.DATASET.STRIDE = 16
cfg.DATASET.RESCALE_TO_BEDLAM = 0.25

cfg.MODEL = CN()
cfg.MODEL.PRETRAINED = None
cfg.MODEL.REG_LAYER = 2
cfg.MODEL.DROP_PATH = 0.3
cfg.MODEL.MOTION_MODULE = False
cfg.MODEL.MOTION_DIM = 384
cfg.MODEL.MLP_DIM=2048
cfg.MODEL.TRANSFORMER = 'oneway'
cfg.MODEL.INVERSE_DEPTH = False
cfg.MODEL.USE_PROMPT = True
cfg.MODEL.TEXT_PROMPT = 'auxiliary'
cfg.MODEL.TEXT_ENCODER = 'CLIP'
cfg.MODEL.CLIP_DROPOUT = 0.0
cfg.MODEL.PROMPT_TYPE = 'prompt'
cfg.MODEL.CAM_ENCODER = 'ray'
cfg.MODEL.TRANSL = 'transl'
cfg.MODEL.MASK_PROMPT = False

cfg.LOSS = CN()



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    cfg.cfg_file = cfg_file
    return cfg.clone()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    cfg_file = args.cfg
    if cfg_file is not None:
        cfg = update_cfg(cfg_file)
    else:
        cfg = get_cfg_defaults()

    return cfg


def prepare_output_dir(cfg):
    logdir = os.path.join(cfg.LOGDIR, cfg.EXP_NAME)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg.cfg_file, dst=logdir + '/config.yaml')

    return cfg
