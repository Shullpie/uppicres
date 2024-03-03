
import argparse
import yaml
import model.data as data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-options', type=str, required=True, help='Path to options file.')
    args = parser.parse_args()
    option_path = args.options

    with open(option_path, 'r') as file_option:
        options = yaml.safe_load(file_option)

    tr, te = data._create_dataset(options)
#     model = BatchSegUnet(options["nns"]["seg"]["batch_seg_unet"])
#     optimizer = optimizers.get_optimizer(model.parameters(), 
#                                          options["nns"]["seg"]["batch_seg_unet"]["optimizer"])
#     scheduler = schedulers.get_scheduler(optimizer, 
#                                          options["nns"]["seg"]["batch_seg_unet"]["scheduler"],
#                                          total_iters=10)
#     trl, trs = data.get_dataloaders(options["dataset"]["seg_dataset"])
#     print(model)
#     print(optimizer)
#     print(scheduler)
#     data.show_batch(next(iter(trl)), options["dataset"]["seg_dataset"]["dataloader"])
#     data.show_batch(next(iter(trl)), options["dataset"]["seg_dataset"]["dataloader"])

if __name__=="__main__":
    main()

    # loss_fn = BCELoss()
    # bm = BaseModel(model, optimizer, scheduler, loss_fn)
    # trainer = pl.Trainer(devices=1,
    #                     accelerator="gpu",
    #                     max_epochs=1,
    #                     logger=False,
    #                     enable_checkpointing=False,
    #                     check_val_every_n_epoch=10)
    # trainer.fit(bm, trl, trs)