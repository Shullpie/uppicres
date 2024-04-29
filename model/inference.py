from model.data.processing import functional
from model.modules import models
import torchvision.transforms as T
from PIL import Image
import torch


@torch.inference_mode()
def infernce(user_img: Image.Image, inference_options: dict, todo: dict[str, bool]):
    user_img = T.ToTensor()(user_img)
    device = inference_options.get('device', 'cpu')
    
    if todo['S']:
        flipped_image = T.RandomHorizontalFlip(1)(user_img)

        user_img, dims = functional.prepare_img(img=user_img, 
                                                mean=inference_options['seg_normalize']['mean'],
                                                std=inference_options['seg_normalize']['std'],
                                                crop=inference_options['crop'])
        
        flipped_image, _ = functional.prepare_img(img=flipped_image, 
                                                  mean=inference_options['seg_normalize']['mean'],
                                                  std=inference_options['seg_normalize']['std'],
                                                  crop=inference_options['crop'])

        model = models.get_inference_model(inference_options=inference_options, device=device, label='S')

        res = []
        step = 4
        n_patches = len(user_img)
        
        for img in [user_img, flipped_image]:
            mask = []
            for i in range(0, n_patches, step):
                if i+step > n_patches:
                    tmp = img[i:].to(device)
                    
                    for patch in model(tmp):
                        mask.append(patch.cpu())
                        print(patch)
                    break

                tmp = img[i:i+step].to(device)
                for patch in model(tmp):
                    mask.append(patch.cpu())
            res.append(torch.stack(mask))
        return res, dims        
        # mask = functional.concatenate_patches(res[0])
        # return mask
        del flipped_image, model, step, n_patches
