from typing import NamedTuple
from torch.utils.data import Dataset


class Datasets(NamedTuple):
    train_set: Dataset
    test_set: Dataset


def  create_datasets(main_options: dict) -> Datasets:
    task = main_options["task"]
    if task == "seg":
        from model.data.datasets.seg_dataset import SegDataSet as DS
    elif task == "clr":
        from model.data.datasets.clr_dataset import ClrDataSet as DS
    else:
        raise NotImplementedError(f"Task {task} is not recognized.")

    train_set = DS(dataset_type_options=main_options['datasets'][f'{task}_dataset']['train'],
                   crop=main_options['crop'])
    test_set = DS(main_options['datasets'][f'{task}_dataset']["test"],
                  crop=main_options['crop'])

    return Datasets(train_set, test_set)