import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

def _create_dataset(dataset_option:dict):
    task = dataset_option["task"]
    if task == "seg":
        from data.seg_dataset import SegDataSet as DS
    elif task == "clear":
        pass #TODO Rewrite when clear model added
    else: 
        raise NotImplementedError(f"Task {task} is not recognized.")
    
    train_set = DS(dataset_option["train"])
    test_set = DS(dataset_option["test"])

    return train_set, test_set

def get_dataloaders(dataset_option:dict):
    train_set, test_set = _create_dataset(dataset_option)
    return (DataLoader(train_set, **dataset_option["dataloader"]),
            DataLoader(test_set, **dataset_option["dataloader"]))
        
def show_batch(batch, dataloader_option):
    b_s = dataloader_option["batch_size"]
    fig, ax = plt.subplots(b_s, 2)
    fig.set_size_inches(8, b_s*4)
    for i in range(b_s):
        ax[i, 0].imshow(ToPILImage()(batch[0][i]))
        ax[i, 0].axes.xaxis.set_visible(False)
        ax[i, 0].axes.yaxis.set_visible(False)

        ax[i, 1].imshow(ToPILImage()(batch[1][i]))
        ax[i, 1].axes.xaxis.set_visible(False)
        ax[i, 1].axes.yaxis.set_visible(False)

    plt.show()    

# c = 2
# for mask in os.listdir(m):
#     p = Image.open(m+mask).convert("L")
#     t_p = T.ToTensor()(p)
#     print(len(t.unique(t_p)))
#     # if (len(t.unique(t_p)) != 2):
#     #     if t.sum(t_p)/(1024*1024) > 0.5:
#     #         t_p = T.v2.v2.RandomInvert(p=1)(t_p)
#     #     for row in range(len(t_p[0])):
#     #         for num in range(len(t_p[0][0])):
#     #             nu = t_p[0][row][num]
#     #             if nu not in [0, 1]:
#     #                 t_p[0][row][num] = 1

#     #     print((t.unique(t_p)))                
#     #     n_i = T.ToPILImage()(t_p)
#     #     n_p = r"model/data/datasets/find_damage_data/n/"
#     #     n_i = n_i.save(n_p+mask, format='PNG')
        