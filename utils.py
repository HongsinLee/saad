import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import random
from math import sqrt
from robustbench.utils import load_model
import numpy as np
import wandb
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os


def samplewise_entropy(logits):
    """
    logits: [B, num_classes]
    return: Tensor shape [B], each sample's entropy
    """
    probs = F.softmax(logits, dim=1)  # [B, C]
    log_probs = F.log_softmax(logits, dim=1)  # [B, C]
    ent_per = -(probs * log_probs).sum(dim=1)  # [B]
    return ent_per

def samplewise_kl_div(input_logits, target_logits):
    """
    KL( target || input ) samplewise
    input_logits: Student logit => logQ
    target_logits: Teacher logit => P
    return: shape [B]
    """
    # P = softmax(target)
    # logQ = log_softmax(input)
    # kl = sum( P*(logP - logQ) )
    with torch.no_grad():
        p = F.softmax(target_logits, dim=1)  # [B, C]
    logq = F.log_softmax(input_logits, dim=1)  # [B, C]
    # reduction='none' => shape [B, C]
    kl_per = F.kl_div(logq, p, reduction='none').sum(dim=1)  # [B]
    return kl_per

def samplewise_renyi_entropy(logits, alpha=2.0, eps=1e-12):
    """
    logits : [B, C]
    return : Tensor [B] – Rényi entropy H_α(p)
    """
    p = F.softmax(logits, dim=1)          # 확률
    if abs(alpha - 1.0) < 1e-6:           # α≃1 ⇒ Shannon
        return -(p * torch.log(p + eps)).sum(dim=1)

    log_sum = torch.log((p ** alpha).sum(dim=1) + eps)   # log Σ p_c^α
    return log_sum / (1.0 - alpha)



def load_dataset(dataset, batch_size):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(
        root='../dataset/', train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root='../dataset/', train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root='../dataset/', train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR100(
            root='../dataset/', train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    elif dataset == 'tinyimg':
        class TinyImageNet(Dataset):
            def __init__(self, dataset_type, transform=None):
                self.root = "../dataset/tiny-imagenet-200/"
                data_path = os.path.join(self.root, dataset_type)

                self.dataset = torchvision.datasets.ImageFolder(root=data_path)

                self.transform = transform

            def __getitem__(self, index):
                img, targets = self.dataset[index]

                if self.transform is not None:
                    img = self.transform(img)

                return img, targets

            def __len__(self):
                return self.dataset.__len__()
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = TinyImageNet("train", transform_train)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        testset = TinyImageNet("val", transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    elif dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.SVHN(root='../dataset/', split='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.SVHN(root='../dataset/', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def load_student(studnet_name, dataset, depth=32, widen_factor=10):
    if dataset == "cifar100":
        from cifar100_models import mobilenet_v2, resnet18, wide_resnet, preresnet
        if studnet_name == "RES-18":
            student = resnet18()
            student = torch.nn.DataParallel(student)
            student = student.cuda()
        elif studnet_name == "MN-V2":
            student = mobilenet_v2()
            student = torch.nn.DataParallel(student)
            student = student.cuda()
        elif studnet_name == "WRN":
            student = wideresnet(depth=depth, num_classes=100, widen_factor=widen_factor, dropRate=0.0)
            student = torch.nn.DataParallel(student)
            student = student.cuda()
    elif dataset == "cifar10":
        from cifar10_models import mobilenet_v2, resnet18, wideresnet
        if studnet_name == "RES-18":
            student = resnet18()
            student = torch.nn.DataParallel(student)
            student = student.cuda()
        elif studnet_name == "MN-V2":
            student = mobilenet_v2()
            student = torch.nn.DataParallel(student)
            student = student.cuda()
        elif studnet_name == "WRN":
            student = wideresnet(depth=depth, num_classes=10, widen_factor=widen_factor, dropRate=0.0)
            student = torch.nn.DataParallel(student)
            student = student.cuda()
    elif dataset == "tinyimg":
        from cifar100_models import pResNet18
        if studnet_name != "RES-18":
            raise AssertionError("Only PreActResNet-18 student for TinyImagenet")
        student = pResNet18(num_classes=200)
        student = torch.nn.DataParallel(student)
        student = student.cuda()
    elif dataset == 'svhn':
        from cifar10_models import mobilenet_v2, resnet18, wideresnet
        if studnet_name == "RES-18":
            student = resnet18()
            student = torch.nn.DataParallel(student)
            student = student.cuda()
        elif studnet_name== "MN-V2":
            student = mobilenet_v2()
            student = torch.nn.DataParallel(student)
            student = student.cuda()
        elif studnet_name == "WRN":
            student = wideresnet(depth=depth, num_classes=10, widen_factor=widen_factor, dropRate=0.0)
            student = torch.nn.DataParallel(student)
            student = student.cuda()
    return student



def load_teacher(teacher_name, dataset):
    if dataset == "cifar10" or dataset == "cifar100":
        teacher = load_model(model_name=teacher_name, dataset=dataset, threat_model='Linf')
    elif dataset == 'tinyimg':
        from cifar100_models import ti_wideresnetwithswish
        teacher = ti_wideresnetwithswish(num_classes=200)
        teacher = torch.nn.Sequential(teacher)
        teacher = torch.nn.DataParallel(teacher)
        checkpoint = torch.load('models/tiny_linf_wrn28-10.pt')
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher = teacher.cuda()
        teacher.eval()
    elif dataset == 'svhn':
        from cifar100_models import svhn_wideresnetwithswish
        teacher = svhn_wideresnetwithswish(num_classes=10)
        teacher = torch.nn.Sequential(teacher)
        teacher = torch.nn.DataParallel(teacher)
        checkpoint = torch.load('models/svhn_linf_wrn28-10.pt')
        teacher.load_state_dict(checkpoint['model_state_dict'])
        teacher = teacher.cuda()
        teacher.eval()
    return teacher





