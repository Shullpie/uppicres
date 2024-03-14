from model.data.dataloaders import dataloaders
from model.modules.models.seg_model import SegModel
import torch

def test(options):
    a = SegModel(options)
    a._save_model_state(10)
