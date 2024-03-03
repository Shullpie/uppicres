from typing import NamedTuple
from torch.utils.data import Dataset


class Datasets(NamedTuple):
    train_set: Dataset
    test_set: Dataset

def create_dataset(dataset_option: dict) -> Datasets:
    task = dataset_option["task"]
    if task == "seg":
        from .seg_dataset import SegDataSet as DS
    elif task == "clear":
        pass  #TODO Rewrite when clear model added
    else: 
        raise NotImplementedError(f"Task {task} is not recognized.")

    train_set = DS(dataset_type_options=dataset_option['datasets'][f'{task}_dataset']['train'], 
                   crop=dataset_option['datasets'][f'{task}_dataset']['crop'])
    test_set = DS(dataset_option['datasets'][f'{task}_dataset']["test"],
                  crop=dataset_option['datasets'][f'{task}_dataset']['crop'])

    return Datasets(train_set, test_set)