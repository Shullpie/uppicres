import argparse
import yaml
from data.dataloaders import dataloaders
from modules.metrics import get_loss_func, get_metrics
from data.visualization import show_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-options', type=str, required=True, help='Path to options file.')
    args = parser.parse_args()
    option_path = args.options

    with open(option_path, 'r') as file_option:
        options = yaml.safe_load(file_option)

    train_loader, _ = dataloaders.create_dataloaders(options)
    metrics = get_metrics(options['nns']['seg']['metrics'])
    lf = get_loss_func(options['nns']['seg']['loss_fn'])
    
    print(metrics)
    print(lf)
    print(len(train_loader))
    show_batch(next(iter(train_loader)))


if __name__=="__main__":
    main()
    