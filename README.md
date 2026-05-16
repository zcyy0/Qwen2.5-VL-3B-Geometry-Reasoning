# 📐 Visual Geometry Reasoning using Qwen2.5-VL with GRPO and SFT

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
- Prompt the model to output solution in \<think>...\</think>\<answer>...\</answer> format
- Evaluated baseline model's accuracy on both validation data and test data.

Results:
- Validation data: Overall Accuracy: 22.1%; Parse Success Rate: 83.5%
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

### 1. Geometry relationship confusion
example question: VWXY is a rhombus. Find angle WXY if angle WVY = 4b+10 and angle XZW = 10b-5
![](./assets/prob_7718.png)

The model correctly recognizes that "because VWXY is a rhombus, all sides are equal" and "The diagonals of a rhombus bisect each other at right angles", but it incorrectly identifies the relationship between angles. It states "angle WVY and angle XZW are complementary angles"


### 3. Theorem misapplication (or blindness)
example question: In triangle PQR, PS=8, QS=14. Find RS.
![](./assets/prob_5734.png)

The model does not know geometric mean theorem and tried to use Pythagorean theorem to solve the question, and got the wrong answer

The failure analysis above shows that the model needs to learn visual grounding and geometry theorems to improve its geometry problem solving ability. SFT is the best option. 

## Stage 2 SFT (Completed)
To address the problems above, I have tried a few of different versions of SFT. The following four version use the same 1500 training examples samples from PGPS9K data. I also performed stratified sampling -- upsampling the problem types that the baseline model performed poorly at in stage 1. 
### Version 1: Long Format SFT
To generate training data, I asked Gemini 2.5 Pro to solve the 1500 problems using the following long format
```
<think>
<facts>
[F1]...
[F2] ...
</facts>
<theorems>
[T1] ...
[T2] ...
</theorems>
<reasoning>
step 1: [T1] ... [F1]
step 2: [F2]
step 3: [T2]
</reasoning>
</think>
<answer></answer>
```
<facts> include the geometry facts provided by PGPS9K. <theorems> include the geometry theorems used to solve the problem. In the reasoning steps, I asked Gemini to cite facts and theorems and intervleave them in the reasoning.

However, after training, the evaluation results on the validation data is only 9%, and the format compliance rate is only 53%. In terms of token length, the average token length of the model output is 3823, median is 888, the max token length is 7,826. Gemini 2.5 Pro's response  average length is 573, median 557, max token length 1068. Upon further investigation, the model's long response is due to outputting a long list of facts in the <facts> section (and a lot of them are incorrect) and not stopping on step 1, step 2,.... until it hits the max token limit. Also, the model did not learn to use facts and theorems in its reasoning.

So this set up has a few drawbacks:
- Gemini's response is too long, much more verbose than Qwen's natural reasoning
- The visual facts is redundant: Qwen does not learn visual grounding from <facts>. Instead, it makes Qwen outputs a long list of visual facts, most of which are not correct.
- Qwen is unable to cite facts and theorems: Qwen may not have enough memory to retain all the facts and theorems. When it's reasoning it casts its attention back the text it has generated, but due to its limited memory it fails to use the facts and theorems.
- Step numbering causes loops in the reasoning: Qwen does not learn when to stop. It keeps outputting step 1, step 2, step 3....

### Version 2: Concise Format SFT
I asked Claude Sonnot to modify Gemini's response: remove <facts><theorems> and <reasoning> blocks. Instead, rewrite in the <think></think><answer></answer> format. Remove step numbering (step 1, step 2...) and replace [F] and [T] tags in the reasoning with the actual facts and theorems. I also asked Claude to write a more concise version of Gemini's response.

After SFT, the accuracy on the validation data is 17.7%, and the format compliance rate is 82.5%. This is still below 21% accuracy of the baseline model. Below is a comparison between the baseline model and the checkpoint
| Metric | Baseline | Version 2 |
|---|---|---|
| **Accuracy** | **21.0%** | 17.7% |
| Answer extracted | 92.8% |82.5% |
| Has `<think>` | 100.0% |100.0% |
| Has `<answer>` | 93.0% | 82.7% |
| Fact recall | 29.1% |27.5% |
| Theorem recall | 9.9% | 12.2% |


| Metric | Baseline | Version 2|
|---|---|---|
| Min | 41 | 31 |
| Max | 7,932 | 8,053 |
| **Mean** | **338** | **1,362** |
| Median | 230 | 195 |

The SFT model has lower fact recall than the baseline. It also suffers from degenerative reasoning loops. If I exclude the degenrative looping outputs, the SFT model has similar accuracy as the baseline model. So the problem is to teach the model to stop reasoning and output answer before it hits max token limit.

### Version 3: Multi-task SFT
To address the limitations above, I also tried multi-task SFT:
- task 1 visual grounding: give the model geometry image and a list of relevant and irrelevant facts, prompt the model to select the relevant facts
- task 2 answer only: give the model the question and the geometry image, ask the model to output answer directly. The purpose is to teach the model to output <answer> tags
- task 3 end-to-end: give the model the image and the question text, prompt the model to output thinking steps and final answer in the \<think>step 1:..., step 2:...\</think>\<answer>...\</answer> format. The model should output visual facts on its own and apply relevant theorems.

The SFT model achieves similar accuracy as the baseline model. Although excluding the loop responses the accuracy is 22%. This is very close to 21% of the baseline model.

### SFT conclusion
It seems SFT makes the model worse than the baseline model. Therefore, it's better to use the baseline model for GRPO.


## Stage 3 GRPO (In Progress)
reward function design: 

## Stage 4 Benchmark Evaluation

