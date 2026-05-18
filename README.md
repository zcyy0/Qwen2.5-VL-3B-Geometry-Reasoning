# 📐 Visual Geometry Reasoning using Qwen2.5-VL

![Status](https://img.shields.io/badge/Status-Training_In_Progress-yellow)
![Model](https://img.shields.io/badge/Base_Model-Qwen_2.5_VL_3B-green)
![Tech](https://img.shields.io/badge/Stack-TRL_%7C_VLLM_%7C_LoRA-blue)

## 📌 Project Overview
This project implements **Group Relative Policy Optimization (GRPO)**, **Supervised Finetuning** and **On Policy Distillation** and investigate if these methods can improve **visual geometry reasoning** in the **Qwen2.5-VL-3B-Instruct** model. 
The training system leverages **HuggingFace TRL** for the RL loop and SFT, **LoRA** for parameter-efficient tuning, and **VLLM** for high-throughput generation during the exploration phase.

**Training Data:** [CASIA-PGPS9K](https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/index.html)

---
## Stage 0 Data Split and processing (Completed)
CASIA-PGPS9K has 9,022 total problems with 30 different problem types. The biggest highlight of this dataset is it includes structural and semantic clauses, which are the extracted geometric properties from the images. This can be very helpful for improving model's visual grounding capability. Some questions share the same geometry images. To avoid data leakage, I used group-level split: all the questions that share the same image belong to the same split. At the same time, I made sure the training, validation and test splits have similar ratios of problem types. The split result is:
- Training data: 7,500 problems
- Validation data: 513 problems 
- Test data: 1,007 problems

I also did additional data processing including the following:
- Some of the original questions use latex expressions. These questions are converted to natural language questions
- The structural and semantic clauses are written in a special annotation. These clauses are converted to functional annotation.
- The ground truth answers in CASIA-PGPS9K are float numbers to three decimals, but the model output answer can be a fractional number or contain symbols such as \sqrt, \pi etc. I wrote a util script that uses latex2sympy2 library to implement a fair comparison between the ground truth answer and the model answer.

## Stage 1 Baseline Evaluation (Completed)
To evaluate the baseline model:
- Give the model the question text and image, and prompt the model to output solution in \<think>...\</think>\<answer>...\</answer> format
- Evaluated baseline model's accuracy on both validation data and test data.

Results:
- Validation data: Overall Accuracy: 22.2%; Parse Success Rate: 83.5%
- Test data: Overall Accuracy: 22.4%; Parse success rate: 83.8%

Model's accuracy broken down by problem type (ordered in ascending order):
| Problem Type | Accuracy |
|---|---|
| Angle Bisector of Triangle | 0% |
| Geometric Mean| 0% |
| Polygon Angle| 0% |
| Circle Chord| 5% |
| Secant Angle  | 6% |
| Secant Segment   | 7% |
|Tangent    | 8% |
|  Inscribed Angle| 10% |
| Rhombus and Square| 14% |
| Median of Triangle| 14% |
| Similarity in Parallel Line| 15% |
| Perpendicular Bisector of Triangle| 20% |
| Isosceles (Equilateral) Triangle| 21% |
| Trigonometry| 22% |
| Parallelogram| 25% |
| Polygon Congruence | 25% |
| Pythagorean Theorem| 25% |
| Midsegment of Triangle| 27% |
| Angle Relation in Triangle| 29% |
| Trapezoid and Kite| 29% |
| Circumference and Area of Circle| 29% |
| Parallel Lines    | 31% |
| Line Segment  | 33% |
| Perimeter and Area of Polygon| 33% |
|  Polygon Similarity| 35% |
| Arc Angle | 36% |
| Perimeter and Area of Quadrangle| 37% |
| Rectangle | 38% |
| Angle| 39% |
| Perimeter and Area of Triangle| 44% |

By looking at individual problems that the model was wrong on, I have found a few failure patterns:

### 1. Visual hallucination
exampl question: BD bisects angle ABC. Find the measure of angle DBC.
![](./assets/prob_3490.png)

The model correctly identifies 2x+7 and 4x-9 in the image, but hallucinate that there is a triangle in the image, and states "The sum of angles in a triangle is 180 degrees. Therefore, angle ABD + angle CBD + angle DBC = 180"

### 2. Geometry relationship confusion
example question: VWXY is a rhombus. Find angle WXY if angle WVY = 4b+10 and angle XZW = 10b-5
![](./assets/prob_7718.png)

The model correctly recognizes that "because VWXY is a rhombus, all sides are equal" and "The diagonals of a rhombus bisect each other at right angles", but it incorrectly identifies the relationship between angles. It states "angle WVY and angle XZW are complementary angles"


### 3. Theorem misapplication (or blindness)
example question: In triangle PQR, PS=8, QS=14. Find RS.
![](./assets/prob_5734.png)

The model does not know geometric mean theorem and tried to use Pythagorean theorem to solve the question, and got the wrong answer

The failure analysis above shows that the model needs to learn visual grounding and geometry theorems to improve its geometry problem solving ability. SFT is the best option. 

To quantify the failure pattern, I have tried the following ablations by varying the prompt:
1. I asked a more capable model to output relevant visual facts in the diagram that help with problem solving, and add these relevant visual facts in the prompt when evaluating the baseline model. The evaluation accuracy on the validation dataset went up from 22% to 30.4%
2. I removed the image, and only provided the model with the question and relevant visual fact texts, and the model's evaluation accuracy is 31.1%, close to the first case
3. I asked a more capable model (Gemini 2.5 Pro) to list the relevant theorems for the problems, and based on 2, added the theorems to the prompt, the model's evaluation accuracy increased to 39.1%. To confirm that the increase in accuracy is due to relevant theorems, not longer prompt, I injected random theorems into the prompt, and the accuracy dropped back to 31.7%
4. By inspecting the failure responses in 3, I found that the model failed to bind objects to the relevant theorems. For example, the model knows Pythagorean Theorem and that a^2 + b^2 = c^2, but it fails to identify the correct sides in the diagram to establish the equation. To verify that this is a bottleneck, I asked a more capable model (Gemini 2.5 Pro) to write object-theorem bindings for the problems, and injected these bindings in the prompt. Based on 3, the accuracy went up to 69.1%.
5. Based on 4, I removed relevant visual facts from the prompt, so the prompt only contained question text + relevant theorems + object-theorem bindings, and the acurracy dropped back to 50%. 
6. By inspecting the failure responses from 4, the remaining 30% incorrect responses are due to loop responses, incorrect arithetic operations.
The results are summarized in the chart below:

| Prompt | Accuracy on validation dataset |
|---|---|
| Question + Image | 22.2% |
| Question + Image + Relevant visual facts | 30.4% |
| Question + Relevant visual facts| 31.1% |
| Question + Relevant visual facts + Relevant theorems | 39.1% |
| Question + Relevant visual facts + Relevant theorems + Object-theoream Bindings | 69.1% |
| Question + + Relevant theorems + Object-theoream Bindings | 49.4% |
| Question + + Relevant theorems + Object-theoream Bindings | 50% |

Based on the findings above, the biggest bottleneck identified is object-theorem binding. And the next bottleneck is extracting useful visual facts from the diagram.

## Stage 2 SFT (Completed)
To address the problems above, I have tried a few of different versions of SFT. The following versions use the same 1500 training examples samples from PGPS9K data. I also performed stratified sampling -- upsampling the problem types that the baseline model performed poorly at in stage 1. 

### Version 1.1: Teacher model's response
I used Gemini 2.5 Pro to write solutions for the problems and explicitly mentioned visual facts and relevant theorems in the solution. However, Gemini 2.5 Pro's response was too verbose, and SFT using these responses caused severe degeneration in the model's output. I then asked Claude Sonnet to re-write the responses to be more concise but still maintained the key visual facts and theorems. After SFT, the accuracy on the validation data is 17.7%, and the format compliance rate is 82.5%. This is still below 21% accuracy of the baseline model. Below is a comparison between the baseline model and the checkpoint
| Metric | Baseline | Version 1 |
|---|---|---|
| **Accuracy** | 22.2% | 17.7% |
| Answer extracted | 92.8% |82.5% |
| Has `<think>` | 100.0% |100.0% |
| Has `<answer>` | 93.0% | 82.7% |
| Min token length | 41 | 31 |
| Max token length | 7,932 | 8,053 |
| Mean token length | 338 | 1,362 |
| Median token length | 230 | 195 |

The model suffers from degenerative reasoning loops. The possible reason is the distribution of the teacher model's response tokens are very different from Qwen 2.5 VL model, and SFT causes a shift in model's output distribution. 

### Version 1.2: Teacher correcting student's response
Based on the analysis from Version 1.1, I let the Qwen model write the response first, and then asked Claude Sonnet to correct the response. The goal is to maintain the style of the model's original response but at the same correct key errors in the reasoning. However, this method did not improve the model's accuracy at all. Upon further investigation, the beginning 1-2 sentences are usually fine, but Claude Sonnet began to make big corrections in the rest of the reasoning, and this still caused changes that are very different from the model's original style. 

### Version 2.1: Rejection-Sampling Fine-tuning
Based on Version 1.1 and Version 1.2, the Qwen model does not learn well from teacher generated responses. So I tried to ask the model to generate the responses itself, and kept the correct ones, and SFT using the data. This increased the accuracy on the validation dataset by 1%. 

### Verstion 2.2: Hint-Augmented Rejection-Sampling Fine-tuning
Version 2.1 has its limitations, it only reinforced the model's correct responses, but if the model does not have the object-theorem binding capability, it will not surface. Another idea I tried based on Version 2.1 is to inject the object-theorem binding in the prompt, and ask the model to generate reasoning response that explicitly mentions the object-theorem binding, and select the correct responses, and SFT using these responses. However, the accuracy on the validation data is 22.4%, very close to the baseline model. 

### SFT conclusion
None of the SFT methods above improved the model, so for GRPO, I decided to use the baseline model. 

## Stage 3 GRPO (Completed)
### GRPO set up
- Curriculum learning: Ask the baseline model to generate 4 rollouts per problem with temperature=0.6. If there are 3 or 4 correct rollouts, classify the problem as easy; if there are 1 to 2 correct rollouts, classify the problem as medium; if there are 0 correct rollouts, classify the problem as hard. The training is divided into 3 stages: 1 easy -> 2 medium -> 3 hard. I further divided the hard problems into stage 3A and stage 3B. Let the model generate 8 rollouts per hard problem, if there's at least 1 correct rollout, the problem belongs to stage 3A. If there is zero correct rollouts, the problems belongs to stage 3B. 
- Number of rollouts K = 8
- One epoch each stage: After running for more than one epoch, the reward and accuracy did not increase.
- Learning rate: 1e-5
- LoRA: LLM + Vision MLP + Projector
- Beta: Stage 1: 0.02. Stage 2: 0.01. Stage 3: 0.005

### Reward design
1. Stage 1: Answer + Format: reward = 1.0 if answer is correct and the strict format is in the output; 0.0 otherwise.
2. Stage 2: Answer + Format + Theorem + Facts: reward = 1.0 + grounding bonus if answer is correct; 0.0 otherwise. grounding bous = w_1 * visual_facts_coverage + w_2 * theorem_coverage
3. Stage 3A: Same reward design as Stage 2
4. Stage 3B: Answer + Format + Theorem + Facts + length penalty. Reward = 1.0 * answer_correct + w_1 * visual_facts_coverage + w_2 * theorem_coverage - 0.2 * loop_or_too_long

### Training analysis in each stage
For each stage, I use Weights and Bias to track the reward mean, reward standard deviation, loss, KL, entropy. I also tracked two metrics: gen_prompt_correct and gen_overall_correct. Gen_prompt_correct computes the percentage of problems the model has at least one correct
rollout for; gen_overall_correct computes the percentage of model's correct rollouts. 

Stage 1
gen_prompt_correct stay between 95% and 100% over the training. gen_overall_correct increased from 40% to 60%.
| Reward mean | Reward std | Loss |
| :---: | :---: | :---: |
| ![](./assets/stage_1_mean_reward.png) | ![](./assets/stage_1_reward_std.png) | ![](./assets/stage_1_loss.png) |

### Evaluation at each stage
| Stage | Accuracy on Validation data |
|---|---|
| Stage 1 | 23%|
| Stage 2| 26.1%|
| Stage 3A|27.2%|
| Stage 3B|27.2%|

The improved accuracy from Stage 2 to Stage 3A is likely noise, so I picked the best checkpoint from Stage 3A. The evaluation accuracy on the test data is 28.9% compared to the baseline model's 22.4%.

## Stage 4 On-Policy Distillation (In Progress)


