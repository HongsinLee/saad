import os
import torch
from torch.optim.swa_utils import AveragedModel, update_bn
import numpy as np
import time
from status import ProgressBar
from args import create_parser
from autoattack import AutoAttack
from utils import load_dataset, load_teacher, load_student, samplewise_entropy, samplewise_kl_div
from attacks import saad_inner_loss, cw_Linf_attack, FGSM, PGD
try:
    import wandb
except ImportError:
    wandb = None

parser = create_parser()
args = parser.parse_known_args()[0]
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

if not args.nowand:
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               config=vars(args), name=args.wandb_name, tags=[args.wandb_tags])
    args.wandb_url = wandb.run.get_url()

trainloader, testloader = load_dataset(args.dataset, args.batch)
student = load_student(args.student, args.dataset, args.depth, args.widen_factor).cuda()
student.train()

teacher = load_teacher(args.teacher_name, args.dataset).cuda()
teacher.eval()

optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
progress_bar = ProgressBar()
swa_model = AveragedModel(student)
swa_start = int(args.swa_epoch)


# ---------- Train -----------
for epoch in range(1, args.epochs + 1):

    for step, (X, y) in enumerate(trainloader):
        student.train()
        X = X.cuda().float()
        y = y.cuda()

        with torch.no_grad():
            teacher_clean = teacher(X)

        inputs_adv = saad_inner_loss(student,teacher,X,y,optimizer,step_size=2/255.0,epsilon=8/255.0,perturb_steps=10, beta=args.lambda_inner)

        optimizer.zero_grad()
        with torch.no_grad():
            delta = inputs_adv - X
            teacher_plus = teacher(inputs_adv)
            teacher_minus = teacher(X - delta)

        student_plus = student(inputs_adv)
        student_minus = student(X - delta)
        student_clean = student(X)

        teacher_ent_per = samplewise_entropy(teacher_plus) # shape [B]
        kl_adv_per_sample = samplewise_kl_div(student_plus, teacher_plus) # shape [B]
        kl_clean_per_sample = samplewise_kl_div(student_clean, teacher_clean)

        #args.igdm_alpha = 1 for cifar10, 10 for cifar100, tinyimagenet
        IGDM_loss_per_sample = args.igdm_alpha * (epoch/args.epochs) * samplewise_kl_div(student_plus - student_minus, teacher_plus - teacher_minus) # shape [B]

        
        min_ent = teacher_ent_per.min().item()     
        # defualt args.entropy_scale = 5
        w = args.entropy_scale*(teacher_ent_per - min_ent)  # shape [B] 
        w = torch.clamp(w, min=0.0) 
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)

        weighted_kl = w * ( kl_adv_per_sample +  IGDM_loss_per_sample) + args.beta * (1 - w_norm) * kl_clean_per_sample
        kl_loss = weighted_kl.mean()  # batch mean

        loss = kl_loss
        loss.backward()
        optimizer.step()

        progress_bar.prog(step, len(trainloader), epoch, loss.item())
            # SWA update & evaluate
    if epoch >= swa_start:
        swa_model.update_parameters(student)
        update_bn(trainloader, swa_model, device='cuda')
        if not args.nowand:
            wandb.log({'epoch': epoch})
    else:
        if not args.nowand:
            wandb.log({'epoch': epoch})

    if epoch in [100, 150]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


update_bn(trainloader, swa_model, device='cuda')

torch.cuda.empty_cache()

# ---------- Test -----------
test_accs = []
test_accs_pgd = []
test_accs_cw = []
test_accs_fgsm = []
swa_model.eval()
for step,(X,y) in enumerate(testloader):
    X = X.cuda().float()
    y = y.cuda()
    inputs_cw = cw_Linf_attack(X, y, swa_model, eps=8/255, alpha=2/225)
    inputs_fgsm = FGSM(X, y, swa_model, eps=8/255)
    inputs_pgd = PGD(X, y, swa_model, eps=8/255, alpha=2/225, steps=20, random_start=True)
    swa_model.eval()
    with torch.no_grad():
        logits = swa_model(X)
        logits_cw = swa_model(inputs_cw)
        logits_fgsm = swa_model(inputs_fgsm)
        logits_pgd = swa_model(inputs_pgd)
    
    predictions_pgd = np.argmax(logits_pgd.cpu().detach().numpy(),axis=1)
    predictions_pgd = predictions_pgd - y.cpu().detach().numpy()
    
    predictions_cw = np.argmax(logits_cw.cpu().detach().numpy(),axis=1)
    predictions_cw = predictions_cw - y.cpu().detach().numpy()

    predictions_fgsm = np.argmax(logits_fgsm.cpu().detach().numpy(),axis=1)
    predictions_fgsm = predictions_fgsm - y.cpu().detach().numpy()

    predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
    predictions = predictions - y.cpu().detach().numpy()
    
    test_accs = test_accs + predictions.tolist()
    test_accs_pgd = test_accs_pgd + predictions_pgd.tolist()
    test_accs_cw = test_accs_cw + predictions_cw.tolist()
    test_accs_fgsm = test_accs_fgsm + predictions_fgsm.tolist()

test_accs = np.array(test_accs)
test_accs_adv = np.array(test_accs_pgd)
test_accs_fgsm = np.array(test_accs_fgsm)
test_accs_cw = np.array(test_accs_cw)
test_acc = np.sum(test_accs==0)/len(test_accs)
test_acc_adv = np.sum(test_accs_adv==0)/len(test_accs_adv)
test_acc_fgsm = np.sum(test_accs_fgsm==0)/len(test_accs_fgsm)
test_acc_cw = np.sum(test_accs_cw==0)/len(test_accs_cw)
print('PGD20 acc',test_acc_adv)
print('CW acc',test_acc_cw)
print('FGSM acc',test_acc_fgsm)
if not args.nowand:
    d2={'clean_acc': test_acc, 'robust_acc': test_acc_adv, 'fgsm_acc':test_acc_fgsm, 'cw_acc':test_acc_cw}
    wandb.log(d2)


save_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
save_dir = './result_models/'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, args.wandb_name + save_time + str(args.student) + '.pt')
torch.save(swa_model.state_dict(), save_path)

swa_model.eval()
autoattack = AutoAttack(swa_model, norm='Linf', eps=8/255.0, version='standard')
x_total = [x for (x, y) in testloader]
y_total = [y for (x, y) in testloader]
x_total = torch.cat(x_total, 0)
y_total = torch.cat(y_total, 0)
_, robust_acc = autoattack.run_standard_evaluation(x_total, y_total)
print('final AA',robust_acc)
if not args.nowand:
    AA_d = {'RESULT_AA': robust_acc}
    wandb.log(AA_d)

if not args.nowand:
    wandb.finish()