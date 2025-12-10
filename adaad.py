import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from status import ProgressBar
from args import create_parser
from autoattack import AutoAttack
from utils import *
from attacks import *
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
criterion_kl = nn.KLDivLoss(reduction="batchmean")

# ---------- Train -----------
for epoch in range(1, args.epochs + 1):

    for step, (X, y) in enumerate(trainloader):
        student.train()
        X = X.cuda().float()
        y = y.cuda()
        inputs_adv = adaad_inner_loss(student, teacher, X,step_size=8/255.0, epsilon=2/255.0)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_plus = teacher(inputs_adv)
        student_plus = student(inputs_adv)
        loss = criterion_kl(F.log_softmax(student_plus, dim=1), F.softmax(teacher_plus, dim=1))
        loss.backward()
        optimizer.step()
        progress_bar.prog(step, len(trainloader), epoch, loss.item())

    # ---------- Evaluation per Epoch-----------
    if epoch %1 ==0 :
        test_accs = []
        test_accs_adv = []
        student.eval()
        for step,(X,y) in enumerate(testloader):
            test_pgd_data = PGD(X, y, student, eps=8/255, alpha=2/255, steps=20)
            student.eval()
            with torch.no_grad():
                logits = student(X)
                logits_adv = student(test_pgd_data)
            
            predictions_adv = np.argmax(logits_adv.cpu().detach().numpy(),axis=1)
            predictions_adv = predictions_adv - y.cpu().detach().numpy()
            
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - y.cpu().detach().numpy()
            
            test_accs = test_accs + predictions.tolist()
            test_accs_adv = test_accs_adv + predictions_adv.tolist()
        test_accs = np.array(test_accs)
        test_accs_adv = np.array(test_accs_adv)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        test_acc_adv = np.sum(test_accs_adv==0)/len(test_accs_adv)
        print('PGD20 acc',test_acc_adv)
        if not args.nowand:
            d2={'clean_acc': np.round(test_acc,4), 'PGD20_acc': np.round(test_acc_adv,4), 'epoch' : epoch}
            wandb.log(d2)



    if epoch in [100, 150]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

torch.cuda.empty_cache()

# ---------- Final Evaluation -----------
test_accs = []
test_accs_pgd = []
test_accs_cw = []
test_accs_fgsm = []
student.eval()
for step,(X,y) in enumerate(testloader):
    X = X.cuda().float()
    y = y.cuda()
    inputs_cw = cw_Linf_attack(X, y, student, eps=8/255, alpha=2/255)
    inputs_fgsm = FGSM(X, y, student, eps=8/255)
    inputs_pgd = PGD(X, y, student, eps=8/255, alpha=2/255, steps=20, random_start=True)
    student.eval()
    with torch.no_grad():
        logits = student(X)
        logits_cw = student(inputs_cw)
        logits_fgsm = student(inputs_fgsm)
        logits_pgd = student(inputs_pgd)
    
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
    d2={'clean_acc': test_acc, 'PGD20_acc': test_acc_adv, 'fgsm_acc':test_acc_fgsm, 'cw_acc':test_acc_cw}
    wandb.log(d2)


save_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
save_dir = './result_models/'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, args.wandb_name + save_time + str(args.student) + '.pt')
torch.save(student.state_dict(), save_path)

student.eval()
autoattack = AutoAttack(student, norm='Linf', eps=8/255.0, version='standard')
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