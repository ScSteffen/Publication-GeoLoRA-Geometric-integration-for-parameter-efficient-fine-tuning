import torch
import argparse
from src.lora_VIT.dataset_utils import choose_dataset
import src.lora_VIT.lora_dlrt as lora_dlrt
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
        "--tau",
        type=float,
        default=0.1,
        metavar="TAU",
        help="rank truncation parameter (default: 0.1)",
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
        "--dlrt", type=str, default="1", metavar="DLRT", help="name of the integrator"
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

    if "vit" in args.net_name.lower():
        from src.lora_VIT.models.vit import vit_b32_dlrt

        n_classes = 10
        if "cifar100" in args.dataset_name:
            n_classes = 100
        if "tiny_imagenet" in args.dataset_name:
            n_classes = 200

        model, low_rank_layers = vit_b32_dlrt(r=args.init_r, n_classes=n_classes)
        model = model.to(device)
        params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad == True
            and "lora" not in n.lower()
            or "lora_e" in n.lower()
        ]
        # for n, p in model.named_parameters():
        #    if p.requires_grad == True and "lora" not in n.lower():
        #        print(n)

        # print(
        #    [
        #        n
        #        for n, p in model.named_parameters()
        #        if p.requires_grad == True
        #        and "lora" not in n.lower()
        #        or "lora_e" in n.lower()
        #    ]
        # )

        if args.opt.lower() == "adam":
            optimizer_S = torch.optim.AdamW(
                params, lr=args.lr, weight_decay=args.wd
            )  # ,lr = args.lr,momentum  = args.momentum)
            # scheduler_UV = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_UV, factor=0.1, patience=3)
            scheduler_S = torch.optim.lr_scheduler.ConstantLR(
                optimizer_S, factor=0.8, total_iters=10
            )
        elif args.opt.lower() == "sgd":
            optimizer_S = torch.optim.SGD(
                params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd
            )
            # scheduler_UV = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_UV, factor=0.1, patience=5)
            scheduler_S = torch.optim.lr_scheduler.ConstantLR(
                optimizer_S, factor=0.8, total_iters=10
            )
        # from models.resnet import low_rank_layers
    elif "test" in args.net_name.lower():
        from src.lora_VIT.models.vit import test

        model, low_rank_layers = test(r=args.init_r)
        model = model.to(device)
        params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad == True and "lora" not in n.lower()
        ]
        if args.opt.lower() == "adam":
            optimizer_S = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
            scheduler_S = torch.optim.lr_scheduler.ConstantLR(
                optimizer_S, factor=0.8, total_iters=10
            )
        elif args.opt.lower() == "sgd":
            optimizer_S = torch.optim.SGD(
                params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd
            )
            scheduler_S = torch.optim.lr_scheduler.ConstantLR(
                optimizer_S, factor=0.8, total_iters=10
            )
    # print(f"low rank layers {low_rank_layers}\n")
    print("=" * 100)
    print(f"training with parameters: {args}")
    # train neural network
    t = lora_dlrt.LoraTrainer(model, low_rank_layers, train_loader, val_loader)
    if args.rank_budget > 0:
        t.train_budget(
            args.epochs,
            torch.nn.CrossEntropyLoss(),
            optimizer_S,
            scheduler_S,
            args.lr,
            args.tau,
            coeff_steps=args.coeff_steps,
            args=args,
            rank_budget=args.rank_budget,
        )
    else:
        t.train_classic(
            args.epochs,
            torch.nn.CrossEntropyLoss(),
            optimizer_S,
            scheduler_S,
            args.lr,
            args.tau,
            coeff_steps=args.coeff_steps,
            args=args,
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
