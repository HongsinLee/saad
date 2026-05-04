# Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation

This repository contains the official implementation of the TMLR paper  
**"Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation"**.
[Paper Link](https://arxiv.org/abs/2512.10275) | [OpenReview](https://openreview.net/forum?id=ek45VamPCE)

**TL;DR**

- Stronger robust teachers do not necessarily yield more robust students; the missing factor is **sample-wise adversarial transferability** from student to teacher.
- We propose **SAAD (Sample-wise Adaptive Adversarial Distillation)**, which **up-weights transferable samples and down-weights non-transferable, high-variance samples** using a cheap entropy-based proxy.

<div align="center">
  <img src="figure/intro.png" width="97%" alt="Student AA accuracy vs. Teacher AA accuracy." />
</div>

---

## Follow-up Work

For a theoretical understanding of why robust teachers can fail in adversarial distillation, please see our follow-up work:

**Toward Understanding Adversarial Distillation: Why Robust Teachers Fail**  
Accepted to **ICML 2026**.  
[Paper Link coming soon]


## Installation

All experiments in this repo were run with:

- Python 3.8
- PyTorch 2.4.1

Install the necessary packages using pip:

```bash
pip install torch torchvision numpy wandb
```

For loading pre-trained teacher models, we rely on RobustBench.
Please visit the [RobustBench repository](https://github.com/RobustBench/robustbench) for setup and model downloads.

---

## Supported Methods

This repository includes implementations of the following adversarial distillation methods:

| Method | Paper Title | Reference |
| :--- | :--- | :--- |
| **ARD** | Adversarially Robust Distillation | [Link](https://arxiv.org/abs/1905.09747) |
| **RSLAD** | Revisiting Adversarial Robustness Distillation: Robust Soft Labels Make Student Better | [Link](https://arxiv.org/abs/2108.07969) |
| **AdaAD** | Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation | [Link](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Boosting_Accuracy_and_Robustness_of_Student_Models_via_Adaptive_Adversarial_CVPR_2023_paper.html) |
| **IGDM**  | Indirect Gradient Matching for Adversarial Robust Distillation | [Link](https://arxiv.org/abs/2312.03286) |
| **SAAD (Ours)** | **Sample-wise Adaptive Weighting for Transfer Consistency in Adversarial Distillation** | [Link](https://arxiv.org/abs/2512.10275)  |

---

## Quick Start
Run SAAD with the default configuration:

```bash
bash main.sh
```

## Contact

If you have questions about **SAAD**, please feel free to open a GitHub issue.