import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def FGSM(images, labels, model, eps=8/255, random_start=False):
    model.eval()


    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    adv_images.requires_grad = True
    outputs = model(adv_images)

    cost = loss(outputs, labels)

    grad = torch.autograd.grad(cost, adv_images,
                                retain_graph=False, create_graph=False)[0]

    adv_images = adv_images.detach() + eps*grad.sign()
    delta = torch.clamp(adv_images - images, min= -eps, max= eps)
    adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images



def PGD(images, labels, model, eps=8/255, alpha=2/225, steps=20, random_start=True):
    model.eval()

    images = images.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min= -eps, max= eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    model.train()

    return adv_images




def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_Linf_attack(X, y, model, eps=8/255, alpha=2/255, attack_iters=50, restarts=10):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta += torch.FloatTensor(*delta.shape).uniform_(-eps, eps).cuda()
        delta.data = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -eps, eps)
            d = clamp(d, 0 - X[index[0], :, :, :], 1 - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return X + max_delta


def saad_inner_loss(model,
                                teacher_model,
                                x_natural,
                                y,
                                optimizer,
                                step_size=0.003,
                                epsilon=0.031,
                                perturb_steps=10,
                                beta=1.0):

    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    teacher_model.eval()

    x_temp = x_natural.detach()
    x_temp.requires_grad_(True) 
    teacher_logits = teacher_model(x_temp)  
    B, num_classes = teacher_logits.shape
    teacher_logits_of_y = teacher_logits[torch.arange(B), y]  # [B]
    logit_sum = teacher_logits_of_y.sum()
    teacher_gradient_for_y = torch.autograd.grad(
        outputs=logit_sum,
        inputs=x_temp,
        create_graph=False,
        retain_graph=False  
    )[0]  
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            # delta = x_adv - x_natural
            input_diff = x_adv - x_natural  

            #   f_T^y(x+delta) ≈ f_T^y(x) + ∇_x f_T^y(x) · delta
            # shape [B],  einsum('bchw,bchw->b')
            teacher_logit_correction = torch.einsum(
                'bchw,bchw->b', teacher_gradient_for_y, input_diff
            )
            corrected_teacher_logits = teacher_logits.detach().clone()
            corrected_teacher_logits[torch.arange(B), y] += beta* teacher_logit_correction

            student_logits = model(x_adv)  # [B, num_classes]

            # KL( student_logits || corrected_teacher_logits )
            loss_kl = criterion_kl(
                F.log_softmax(student_logits, dim=1),
                F.softmax(corrected_teacher_logits, dim=1)
            )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(x_adv, requires_grad=False)
    optimizer.zero_grad()
    return x_adv


def adaad_inner_loss(model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    if BN_eval:
        model.eval()

    # set eval mode for teacher model
    teacher_model.eval()
    # generate adversarial example
    if random_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    else:
        x_adv = x_natural.detach()
    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_model(x_adv), dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural -
                          epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    if BN_eval:
        model.train()
    model.train()

    x_adv = Variable(torch.clamp(x_adv, clip_min, clip_max),
                     requires_grad=False)
    return x_adv


def rslad_inner_loss(model,
                teacher_logits,
                x_natural,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    return x_adv
