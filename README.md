# Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation

This repository contains the official code of the paper: **"Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation"**.

In this work, we propose **SAAD (Sample-wise Adaptive Adversarial Distillation)**, a method designed to improve adversarial robustness by adaptively weighting sample-wise transfer consistency. This repository also provides implementations of several state-of-the-art adversarial distillation methods.

<div align="center">
<img src="figure/intro.pdf" width="97%" alt="Student AA accuracy vs. Teacher AA accuracy." />
<p><i>(a) Student AA accuracy vs. Teacher AA accuracy.</i></p>

<img src="figure/intro_architecture.pdf" width="97%" alt="Student AA accuracy vs. Teacher architecture size." />
<p><i>(b) Student AA accuracy vs. Teacher architecture size. Colored regions represent the same architectures.</i></p>

<img src="figure/sorted_intro.pdf" width="97%" alt="Student AA accuracy vs. Fraction of adversarial examples that transfer from student to teacher (TAS)." />
<p><i>(c) Student AA accuracy vs. Fraction of adversarial examples that transfer from student to teacher (TAS).</i></p>

<p><b>Figure 1:</b> Adversarial distillation results on CIFAR-10 with a ResNet-18 student. Detailed teacher information and full experimental results are provided in the paper.</p>
</div>

---

## Installation

This code requires **Python 3.8+** and **PyTorch 2.1.0+**.

### 1. Python Dependencies
Install the necessary packages using pip:

```bash
pip install torch>=2.1.0 torchvision numpy wandb
```

### 2. RobustBench
For loading pre-trained models as teacher model, we rely on **RobustBench**.
Please visit the official repository to download:

* **Official Repository:** [https://github.com/RobustBench/robustbench](https://github.com/RobustBench/robustbench)


---

## Supported Methods

We provide implementations for the following adversarial distillation methods:

| Method | Paper Title | Reference |
| :--- | :--- | :--- |
| **ARD** | Adversarially Robust Distillation | [Paper link](https://arxiv.org/abs/1905.09747) |
| **RSLAD** | Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better | [Paper link](https://arxiv.org/abs/2108.07969) |
| **AdaAD** | Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation | [Paper link](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Boosting_Accuracy_and_Robustness_of_Student_Models_via_Adaptive_Adversarial_CVPR_2023_paper.html) |
| **IGDM** | Indirect Gradient Matching for Adversarial Robust Distillation | **(Ours)** [Paper Link](https://arxiv.org/abs/2312.03286) |
| **SAAD** | **Sample-wise Adaptive Weighting for Transfer Consistency** | **(Ours)** |

---

## Quick Start
To start training with the default configuration (SAAD), simply run:

```bash
bash main.sh
```

## Contact

If you have any questions regarding **IGDM** or **SAAD**, please feel free to open a GitHub issue.