# 📐 Visual Geometry Reasoning using Qwen2.5-VL

![Status](https://img.shields.io/badge/Status-Training_In_Progress-yellow)
![Model](https://img.shields.io/badge/Base_Model-Qwen_2.5_VL_3B-green)
![Tech](https://img.shields.io/badge/Stack-TRL_%7C_VLLM_%7C_LoRA-blue)

This project investigates whether supervised fine-tuning and GRPO can improve visual geometry reasoning in Qwen2.5-VL-3B-Instruct on the [CASIA-PGPS9K](https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/index.html) geometry benchmark.
I built a full training and evaluation pipeline using:

- Hugging Face TRL for SFT and GRPO training
- LoRA for parameter-efficient fine-tuning
- vLLM for high-throughput rollout generation
- A symbolic answer parser for robust comparison between model outputs and numeric ground-truth answers
- A curriculum-based GRPO setup that separates problems by rollout difficulty

The baseline Qwen2.5-VL-3B-Instruct model achieved:
| Split | Accuracy | Parse success rate |
|---|---|---|
| Validation | 22.2% |83.5%|
| Test| 22.4% |83.8%|

The final pipeline improved Qwen2.5-VL-3B-Instruct from 22.4% to 30.69% accuracy on the 1,007-problem CASIA-PGPS9K held-out test set, a +8.29 percentage-point gain corresponding to roughly 83 additional solved problems. The pipeline first used curriculum GRPO, which improved test accuracy to 28.9%, then applied vanilla DPO on HardA rollout pairs to further improve accuracy to 30.69%.

The main finding is that the model’s largest bottleneck is object-theorem binding, not merely theorem recall. Oracle ablations showed that providing object-theorem bindings produced the largest accuracy jump. GRPO was effective on easy and medium curriculum buckets, where the model had mixed correct and incorrect rollouts, but plateaued on harder buckets. DPO was more effective after GRPO because it directly used high-k sampled HardA rollouts to teach the model to prefer correct geometry trajectories over plausible but incorrect ones.

The detailed project report is in the project_report.md file
