import torchvision.transforms as T
import torchvision.datasets as datasets
import torch
import os


def choose_dataset(dataset_name: str, batch_size: int, num_workers: int, size=32):
    """
    selects a dataset by name
    """
    if dataset_name == "cifar10":
        return load_cifar10(batch_size, num_workers, size=size)
    elif dataset_name == "cifar100":
        return load_cifar100(batch_size, num_workers, size=size)
    elif dataset_name == "cifar10_b&w":
        return load_cifar10_bw(batch_size, num_workers)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(batch_size, num_workers)
    elif dataset_name == "svhn":
        return load_svhn(batch_size, num_workers)
    elif dataset_name == "mnist":
        return load_mnist(batch_size, num_workers)
    elif dataset_name == "imagenet":
        return load_imagenet(batch_size, num_workers, dummy=False)
    if dataset_name == "tiny_imagenet":
        return load_tiny_imagenet(batch_size, num_workers, size=size)
    else:
        print("dataset not available. Exiting")
        exit(1)


def load_imagenet(
    batch_size: int, num_workers: int, dummy: bool = False, datapath="./imageNet/"
):
    """
    returns train_loader, val_loader and test_loader for the data set
    params: dummy :=  for debugging purposes, if no imagenet data is available
    """
    data = datapath
    if dummy:
        print("=> Dummy data is used!")
        n_train = 1000  # batchsize
        n_val = 64
        train_dataset = datasets.FakeData(n_train, (3, 224, 224), 1000, T.ToTensor())
        val_dataset = datasets.FakeData(n_val, (3, 224, 224), 1000, T.ToTensor())
    else:
        traindir = os.path.join(
            data, "train"
        )  # you need to store the training data in $IMAGENET_PATH/train
        valdir = os.path.join(
            data, "validation"
        )  # you need to store the training data in $IMAGENET_PATH/validation
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            T.Compose(
                [
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize,
                ]
            ),
        )

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )

    return train_loader, validation_loader, None


def load_cifar10(batch_size: int, num_workers: int, size=32):
    """
    returns train_loader, val_loader and test_loader for the data set
    """
    # transform = T.Compose([T.Resize(size),T.ToTensor()])
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(*stats, inplace=True),
        ]
    )
    valid_transform = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(*stats)])
    train_loader = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_loader = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=valid_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, None


def load_cifar100(batch_size: int, num_workers: int, size=32):
    """
    returns train_loader, val_loader and test_loader for the data set
    """
    # transform = T.Compose([T.Resize(size),T.ToTensor()])
    stats = ((0.5071, 0.4867, 0.4408), (0.2673, 0.2564, 0.2762))
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(*stats, inplace=True),
        ]
    )
    valid_transform = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(*stats)])
    train_loader = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_loader = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=valid_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, None


def load_cifar10_bw(batch_size: int, num_workers: int):
    """
    returns train_loader, val_loader and test_loader for the data set
    """
    transform = T.Compose([T.Grayscale(), T.ToTensor()])
    train_loader = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    val_loader = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader, val_loader, None


def load_fashion_mnist(batch_size: int, num_workers: int):
    """
    returns train_loader, val_loader and test_loader for the data set
    """
    trans = T.Compose([T.ToTensor()])
    train_loader = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=trans
    )
    val_loader = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader, val_loader, None


def load_svhn(batch_size: int, num_workers: int):
    """
    returns train_loader, val_loader and test_loader for the data set
    """

    trans = T.Compose([T.ToTensor()])
    train_loader = datasets.SVHN(root="./data", download=True, transform=trans)
    val_loader = datasets.SVHN(
        root="./data", train=False, download=True, transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return train_loader, val_loader, None


def load_mnist(batch_size: int, num_workers: int):
    """
    returns train_loader, val_loader and test_loader for the data set
    """

    trans = T.Compose([T.ToTensor()])
    train_loader = datasets.MNIST(
        root="./data", train=True, download=True, transform=trans
    )
    val_loader = datasets.MNIST(
        root="./data", train=False, download=True, transform=trans
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, val_loader, None


def load_tiny_imagenet(
    batch_size: int, num_workers: int, dummy: bool = False, size: int = 32
):
    def torch_transforms(is_train=False):
        # Mean and standard deviation of train dataset
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transforms_list = []
        # Use data aug only for train data
        if is_train:
            transforms_list.extend(
                [
                    T.RandomCrop(64, padding=4),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                ]
            )
        transforms_list.extend(
            [
                T.ToTensor(),
                T.Normalize(mean, std),
                T.Resize(size),
            ]
        )
        # if is_train:
        # transforms_list.extend([
        #    transforms.RandomErasing(0.15)
        # ])
        return T.Compose(transforms_list)

    if dummy:
        print("=> Dummy data is used!")
        n_train = 1000  # batchsize
        n_val = 64
        train_dataset = datasets.FakeData(n_train, (3, 64, 64), 200, T.ToTensor())
        val_dataset = datasets.FakeData(n_val, (3, 64, 64), 200, T.ToTensor())
    else:

        traindir = "./data/tiny-imagenet-200/new_train"
        valdir = "./data/tiny-imagenet-200/new_test"
        # train_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
        # val_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')

        train_transforms = torch_transforms(is_train=True)

        val_transforms = torch_transforms(is_train=False)

        train_dataset = datasets.ImageFolder(traindir, train_transforms)

        val_dataset = datasets.ImageFolder(valdir, val_transforms)

    # Create a custom dataset class for Tiny ImageNet
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )

    return train_loader, validation_loader, None
