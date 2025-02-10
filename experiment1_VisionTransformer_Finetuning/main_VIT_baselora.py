import torch
import argparse
from src.lora_VIT.dataset_utils import choose_dataset
from src.lora_VIT.lora_base_trainer_VIT import BaseLoraTrainer
import subprocess


def main():
    ###################### parser creation  ######################
    parser = argparse.ArgumentParser(
        description="Pytorch dlrt training for vgg of imagenet"
    )
    # Arguments for network training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="starting learning rate for optimizers (default: 0.05)",
    )
    parser.add_argument(
        "--start_cr",
        type=float,
        default=0.0,
        metavar="START_CR",
        help="starting layerwise compression (default: 0.0)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="MOM",
        help="momentum (default: 0.05)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        metavar="WD",
        help="Weight decay (default: 0.0005)",
    )
    parser.add_argument(
        "--init_r",
        type=int,
        default=32,
        metavar="INIT_R",
        help="initial lora rank (default 32)",
    )
    parser.add_argument(
        "--net_name",
        type=str,
        default="test",
        metavar="NET_NAME",
        help="name of the network",
    )
    parser.add_argument(
        "--opt", type=str, default="adam", metavar="OPT", help="name of the optimizer"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100", "tiny_imagenet"],
        metavar="DATASET_NAME",
        help="name of the dataset",
    )
    parser.add_argument(
        "--save_weights",
        type=bool,
        default=False,
        metavar="save_weights",
        help="save weights progress",
    )
    parser.add_argument(
        "--wandb", type=int, default=0, help="Activate wandb logging: 0=no, 1=yes"
    )
    parser.add_argument(
        "--adaptivelora",
        type=bool,
        default=True,
        help="adaptive adalora",
    )
    parser.add_argument(
        "--coeff_steps", type=int, default=0, help="Coefficient finetuning steps"
    )

    parser.add_argument(
        "--rank_budget", type=int, default=-1, help="Rank Budget for the whole network"
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = get_available_device()
        print(f"Using Cuda GPU {device}")
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and not force_cpu:
    #     device = torch.device("mps")
    #     print('Using Apple Silicon MPS')
    else:
        device = torch.device("cpu")
        print("Using CPU")
    # Define the hyperparameters

    # -------- Dataset Selection -----------
    train_loader, val_loader, test_loader = choose_dataset(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        num_workers=4,
        size=224 if "vit" in args.net_name.lower() else 32,
    )

    if "vit_adalora" in args.net_name.lower():
        from src.lora_VIT.models.vit import vit_b32_adalora

        args.adaptive_lora = True
        n_classes = 10
        if "cifar100" in args.dataset_name:
            n_classes = 100
        if "tiny_imagenet" in args.dataset_name:
            n_classes = 200

        model, low_rank_layers = vit_b32_adalora(r=args.init_r, n_classes=n_classes)
        model = model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        if args.opt.lower() == "adam":
            optimizer_UV = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
            scheduler_UV = torch.optim.lr_scheduler.ConstantLR(
                optimizer_UV, factor=0.8, total_iters=10
            )
        elif args.opt.lower() == "sgd":
            optimizer_UV = torch.optim.SGD(
                params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd
            )
            scheduler_UV = torch.optim.lr_scheduler.ConstantLR(
                optimizer_UV, factor=0.8, total_iters=10
            )
    elif "vit_lora" in args.net_name.lower():
        from src.lora_VIT.models.vit import vit_b32_lora

        n_classes = 10
        if "cifar100" in args.dataset_name:
            n_classes = 100
        if "tiny_imagenet" in args.dataset_name:
            n_classes = 200

        model = vit_b32_lora(r=args.init_r, n_classes=n_classes)
        model = model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        if args.opt.lower() == "adam":
            optimizer_UV = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
            scheduler_UV = torch.optim.lr_scheduler.ConstantLR(
                optimizer_UV, factor=0.8, total_iters=10
            )
        elif args.opt.lower() == "sgd":
            optimizer_UV = torch.optim.SGD(
                params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd
            )
            scheduler_UV = torch.optim.lr_scheduler.ConstantLR(
                optimizer_UV, factor=0.8, total_iters=10
            )
    else:
        raise NotImplementedError

    print("=" * 100)
    print(f"training with parameters: {args}")
    # train neural network
    t = BaseLoraTrainer(model, low_rank_layers, train_loader, val_loader)
    t.train(
        num_epochs=args.epochs,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer_UV,
        scheduler=scheduler_UV,
        args=args,
        rank_budget=args.rank_budget,
    )
    print("Training finished.")


def get_available_device():
    # Get GPU memory usage using nvidia-smi
    cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    memory_used = subprocess.check_output(cmd.split()).decode().strip().split("\n")
    memory_used = [int(memory.strip()) for memory in memory_used]

    # Find GPU with least memory usage
    device = memory_used.index(min(memory_used))
    return torch.device(f"cuda:{device}")


if __name__ == "__main__":

    main()
