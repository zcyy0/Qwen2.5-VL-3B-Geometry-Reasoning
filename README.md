# 📐 Visual Geometry Reasoning using Qwen2.5-VL
This project investigates whether supervised fine-tuning and GRPO can improve visual geometry reasoning in Qwen2.5-VL-3B-Instruct on the [CASIA-PGPS9K](https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/index.html) geometry benchmark.
I built a full training and evaluation pipeline using:

- Hugging Face TRL for SFT and GRPO training
- LoRA for parameter-efficient fine-tuning
- vLLM for high-throughput rollout generation
- A symbolic answer parser for robust comparison between model outputs and numeric ground-truth answers
- A curriculum-based GRPO setup that separates problems by rollout difficulty
- A DPO experiment as a comparison to GRPO

The baseline Qwen2.5-VL-3B-Instruct model achieved:
| Split | Accuracy | Parse success rate |
|---|---|---|
| Validation | 20.0% |83.5%|
| Test| 21.4% |83.8%|

The experiments below moved the test number from 21.4% to 29.3% accuracy on the 1,007-problem CASIA-PGPS9K held-out test set, a +7.9 percentage-point gain corresponding to roughly 80 additional solved problems. the significant gain comes from Easy+Medium GRPO; hard-bucket stages and DPO did not move the test number.

The main finding is that the model’s largest bottleneck is object-theorem binding, not merely theorem recall. Oracle ablations showed that providing object-theorem bindings produced the largest accuracy jump. GRPO was effective on easy and medium curriculum buckets, but on harder buckets GRPO mostly results in redistribution rather than improving problem solving capability. DPO has similar performance as GRPO on the harder problems.

Reports:
- The detailed report is in the project_report.md file
- I also trained 7B model on 2 GPUs using FSDP Zero3 to study distributed training. The full report and the profiling details are in the FSDP_report.md file 

