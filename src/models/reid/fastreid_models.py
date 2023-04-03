import torch

from collections import OrderedDict

import os.path as osp
import sys
from pathlib import Path


# Enable imports from the fast-reid directory
root = Path(__file__).parent.parent.parent.parent
_FASTREID_ROOT = root / 'fast-reid'
sys.path.append(str(_FASTREID_ROOT))

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.build import build_transforms

_WEIGHTS_DIR = osp.join(root,'fastreid-models')
_FASTREID_MODEL_ZOO = {'msmt_SBS_R101_ibn': ('configs/MSMT17/sbs_R101-ibn.yml', 'model_weights/msmt_sbs_R101-ibn.pth'),
                        'msmt_SBS_S50':('configs/MSMT17/sbs_S50.yml', 'model_weights/msmt_sbs_S50.pth'),
                        'msmt_BOT_R101_ibn': ('configs/MSMT17/bagtricks_R101-ibn.yml', 'model_weights/msmt_bot_R101-ibn.pth'),
                        'msmt_AGW_R101_ibn': ('configs/MSMT17/AGW_R101-ibn.yml', 'model_weights/msmt_agw_R101-ibn.pth'),

                        # More BOT methods on MSMT17
                        'msmt_BOT_S50': ('configs/MSMT17/bagtricks_S50.yml', 'model_weights/msmt_bot_S50.pth'),
                        'msmt_BOT_R50_ibn': ('configs/MSMT17/bagtricks_R50-ibn.yml', 'model_weights/msmt_bot_R50-ibn.pth'),
                        'msmt_BOT_R50': ('configs/MSMT17/bagtricks_R50.yml', 'model_weights/msmt_bot_R50.pth'),

                        # Some BOT methods on Market
                        #https://github.com/JDAI-CV/fast-reid/blob/master/configs/Market1501/bagtricks_R101-ibn.yml
                        'market_BOT_R101_ibn': ('configs/Market1501/bagtricks_R101-ibn.yml', 'model_weights/market_bot_R101-ibn.pth'),
                        'market_BOT_R50_ibn': ('configs/Market1501/bagtricks_R50-ibn.yml', 'model_weights/market_bot_R50-ibn.pth'),
                        
                        'market_bot_S50': ('configs/Market1501/bagtricks_S50.yml', 'model_weights/market_bot_S50.pth'),
            
                        # MGN
                        'market_mgn_R50_ibn': ('configs/Market1501/mgn_R50-ibn.yml', 'model_weights/market_mgn_R50-ibn.pth')
                        
                        }            


print(_WEIGHTS_DIR)

def _get_cfg(fastreid_cfg_file, fastreid_model_weights):
    """
    Create configs and perform basic setups.
    """    
    args = default_argument_parser().parse_args(['--config-file', osp.join(_FASTREID_ROOT, fastreid_cfg_file), '--num-gpus', '1', 'MODEL.WEIGHTS', osp.join(_WEIGHTS_DIR, fastreid_model_weights)])
    #cfg = setup(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False

    return cfg

def _load_ckpt(model, cfg):
    # Fix keys mismatches due to using different versions!
    ckpt=torch.load(cfg.MODEL.WEIGHTS, map_location = torch.device('cuda'))
    state_dict = model.state_dict()
    new_ckpt_model = OrderedDict()
    resave_ckpt=False
    for key, val in ckpt['model'].items():
        if key.startswith('heads.bnneck'):
            key_ = key.replace('heads.bnneck', 'heads.bottleneck.0')
            assert key_ in state_dict
            resave_ckpt=True
            
        else:
            key_ = key
        
        new_ckpt_model[key_] = val

    ckpt['model'] = new_ckpt_model
    if resave_ckpt:
        torch.save(ckpt, cfg.MODEL.WEIGHTS)
    
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

    return model


#def get_mo
def load_fastreid_model(reid_arch_name):
    #model_name = self.config.reid_arch.split('fastreid_')[-1]
    model_name = reid_arch_name.split('fastreid_')[-1]
    fastreid_cfg_file, fastreid_model_weights = _FASTREID_MODEL_ZOO[model_name]

    cfg = _get_cfg(fastreid_cfg_file, fastreid_model_weights)
    model = DefaultTrainer.build_model(cfg)

    model = _load_ckpt(model, cfg)

    transforms = build_transforms(cfg, is_train=False)
    feature_embedding_model = model.eval()

    return feature_embedding_model, transforms