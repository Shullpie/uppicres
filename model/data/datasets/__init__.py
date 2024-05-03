from typing import NamedTuple
from torch.utils.data import Dataset


class Datasets(NamedTuple):
    train_set: Dataset
    test_set: Dataset


def  create_datasets(options: dict) -> Datasets:
    task = options["task"]
    if task == "seg":
        from model.data.datasets.seg_dataset import SegDataSet as DS
    elif task == "clr":
        from model.data.datasets.clr_dataset import ClrDataSet as DS
    else:
        raise NotImplementedError(f"Task {task} is not recognized.")

    train_set = DS(options=options, mode='train')
    test_set = DS(options=options, mode='test')

    return Datasets(train_set, test_set)