GPU_ID=2

dataset=cifar10
teacher_name=Gowal2021Improving_28_10_ddpm_100m  
student=RES-18

method=saad
igdm_alpha=1
beta=0

nowand=1 #set 0 to use wandb
wandb_entity=your_wandb_entity
wandb_project=your_wandb_project

seed=0
epochs=200

CUDA_VISIBLE_DEVICES=$GPU_ID python ${method}.py --wandb_name ${method}_${beta}_${igdm_alpha}_${seed}_${epochs}_${teacher} --beta $beta --igdm_alpha $igdm_alpha --nowand $nowand --wandb_project $wandb_project --wandb_entity $wandb_entity --method $method --epochs $epochs --teacher_name $teacher_name --student $student --dataset $dataset --seed $seed

# Teacher example for cifar10
# Bartoldson2024Adversarial_WRN-94-16 ,  Gowal2021Improving_28_10_ddpm_100m  


