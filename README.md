# 📐 Visual Geometry Reasoning using Qwen2.5-VL

![Status](https://img.shields.io/badge/Status-Training_In_Progress-yellow)
![Model](https://img.shields.io/badge/Base_Model-Qwen_2.5_VL_3B-green)
![Tech](https://img.shields.io/badge/Stack-TRL_%7C_VLLM_%7C_LoRA-blue)

## 📌 Project Summary
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

The best GRPO checkpoint improved held-out test accuracy from 22.4% to 28.9% and parse success rate from 83.5% to 90.86% on 1,007 untouched test problems, a +6.5 percentage-point gain on accuracy, corresponding to roughly 65 additional solved problems.

A rough two-proportion test gives z≈3.35, suggesting that this held-out test improvement is unlikely to be explained by sampling noise alone. A paired gain/loss analysis would be the preferred follow-up test if per-example predictions are available.

The main research finding is that the largest bottleneck is not just theorem knowledge. The biggest bottleneck appears to be object-theorem binding: the model often knows the relevant theorem but fails to bind it to the correct points, segments, angles, or circles in the diagram.

A second key finding is that GRPO works best when the model already has mixed correct and incorrect rollouts. It helped on easier and medium-difficulty curriculum buckets, but showed little additional improvement on the hardest buckets. High pass@k on HardA examples showed that the model can sometimes sample correct solutions, but GRPO did not reliably convert those sampled tail solutions into higher accuracy@1.

## Dataset and Evaluation Setup
### Dataset
The project uses CASIA-PGPS9K, a visual geometry problem dataset with approximately 9K geometry problems across 30 problem types.

One useful feature of this dataset is that it includes structural and semantic clauses describing the geometry diagram, such as:
```
length(AB) = 8
angle(ABC) = 90
perpendicular(AB, CD)
collinear(A, B, C)
midpoint(M, AB)
```
These annotations are not given to the model at final evaluation, but they are useful for reward design and diagnostic analysis.

### Data Split
Some questions share the same geometry image. To avoid image-level leakage, I used a group-level split: all questions sharing the same image are assigned to the same split.

I also stratified the split to keep problem-type distributions similar across train, validation, and test.
| Split | Number of Problems | 
|---|---|
| Training | 7500 |
| Validation | 514 |
| Test| 1007 |

The test set was held out during prompt engineering, curriculum mining, checkpoint selection, and hyperparameter tuning. Validation was used for checkpoint selection and analysis.

### Data Processing
I performed several preprocessing steps:

- Converted some original LaTeX-style questions into natural language.
- Converted PGPS9K structural and semantic clauses into a normalized functional annotation format.
- Built a symbolic answer parser so the model can output decimals, fractions, radicals, π-expressions, units, or simple equations while still being fairly compared against the numeric ground truth.


## Evaluation Protocol
All reported main accuracy numbers are accuracy@1.

Evaluation uses:
```
Input: image + question only
Output format: <think>...</think><answer>...</answer>
Decoding: greedy
Seed: fixed, SamplingParams(seed=42)
```
No visual facts, theorem labels, object-theorem bindings, or PGPS9K clauses are included in the evaluation prompt.

Evaluation and training use the same base prompt format.

### Answer Extraction and Checking
The evaluation pipeline is:
```
extract_answer → parse_numeric → check_answer
```

1. extract_answer: The evaluator extracts the substring inside the first valid <answer>...</answer> block. Anything outside the first <answer> block is ignored.
2. parse_numeric: The parser reduces many model output formats to floats, including:
```
\frac{1}{2}
\sqrt{3}
2\pi
3 + \sqrt{5}
30°
2.5 cm
x = 7
≈ 12.6
```
If the parser extracts multiple comma-separated values, such as:
```
x = 4, y = 7
```
the prediction is counted correct if any extracted value matches the gold answer within tolerance.

3. check_answer
The answer is considered correct if ∣pred−gold∣<0.01.

4. The following cases are counted as wrong:
```
No <answer> tag
Unclosed <answer> tag
No parseable answer inside <answer>
Parsed answer differs from gold by ≥ 0.01
```

## Baseline Results
The baseline Qwen2.5-VL-3B-Instruct model was evaluated with image + question only.
| Split | Accuracy | Parse success rate |
|---|---|---|
| Validation | 22.2% |83.5%|
| Test| 22.4% |83.8%|

### Baseline Accuracy by Problem Type
The baseline model has very uneven performance across geometry types. Some theorem families are nearly unsolved, while simpler area, angle, and perimeter problems are much more tractable.
| Problem Type | Accuracy |Correct/Total|
|---|---|---|
| Angle Bisector of Triangle | 0% |0/6|
| Geometric Mean| 0% | 0/10|
| Polygon Angle| 0% | 0/11|
| Secant Angle  | 5.9% | 1/17|
| Secant Segment   | 6.7% |1/15|
| Circle Chord| 7.5% |3/40|
| Inscribed Angle| 10% |2/20|
| Median of Triangle| 14.3% |2/14|
| Similarity in Parallel Line| 15.4% |2/13|
| Tangent    | 16.7% |2/12|
| Perimeter and Area of Polygon| 16.7% |1/6|
| Rhombus and Square| 18.2% |4/22|
| Parallel Lines    | 18.8% |6/32|
| Trapezoid and Kite| 21.4% |6/28|
|Perimeter and Area of Triangle|22.2%|2/9|
| Line Segment  | 22.2% |2/9|
| Circumference and Area of Circle| 23.5% |8/34|
| Polygon Congruence | 25% | 3/12|
| Pythagorean Theorem| 25% |2/8|
| Midsegment of Triangle| 26.7% |4/15|
| Trigonometry| 28.1% |9/32|
| Angle Relation in Triangle| 28.6% |4/14|
| Isosceles (Equilateral) Triangle| 31.6% |6/19|
|  Polygon Similarity| 35% |7/20|
| Arc Angle | 35.7% |5/14|
| Parallelogram| 37.5% |9/24|
| Rectangle | 37.5% |3/8|
| Angle| 38.9% |7/18|
| Perpendicular Bisector of Triangle| 40% |2/5|
| Perimeter and Area of Quadrangle| 40.7% |11/27|

## Baseline Failure Analysis
#### Visual Hallucination
Example question:
> BD bisects angle ABC. Find the measure of angle DBC.
> ![](./assets/prob_3490.png){width=100px height=100px}
The model correctly identifies expressions such as 2x+7 and 4x−9, but hallucinates a triangle relation that is not present in the diagram. It states:
> “The sum of angles in a triangle is 180 degrees. Therefore, angle ABD + angle CBD + angle DBC = 180.”
This suggests the model sometimes recognizes local visual text but fails to ground the overall geometric structure.

#### Geometry Relationship Confusion
Example question:
>VWXY is a rhombus. Find angle WXY if angle WVY = 4b + 10 and angle XZW = 10b - 5.
>![](./assets/prob_7718.png){width=100px height=100px}

The model recognizes relevant properties of a rhombus:
```
All sides are equal.
The diagonals of a rhombus bisect each other at right angles.
```
However, it incorrectly states that angle WVY and angle XZW are complementary angles.

This is not a pure theorem-recall failure. The model knows some rhombus facts, but applies the wrong relationship to the wrong diagram objects.

#### Theorem Blindness
Example question:
> In triangle PQR, PS = 8, QS = 14. Find RS
> ![](./assets/prob_5734.png)
The model fails to use the geometric mean theorem and instead tries to apply the Pythagorean theorem, producing an incorrect answer.

This suggests that the model needs stronger theorem selection and theorem-application ability.

## Diagnostic Oracle Ablations
To better understand the bottlenecks, I ran oracle-assisted prompt ablations. These are diagnostic upper-bound experiments, not deployable end-to-end evaluations. They use information generated by stronger models or dataset annotations to identify which missing capabilities most constrain the 3B model.

| Prompt | Accuracy on validation dataset |
|---|---|
| Question + Image | 22.2% |
| Question + Image + Relevant visual facts | 30.4% |
| Question + Relevant visual facts| 31.1% |
| Question + Relevant visual facts + Relevant theorems | 39.1% |
| Question + Relevant visual facts + Relevant theorems + Object-theoream Bindings | 69.1% |
| Question + + Relevant theorems + Object-theoream Bindings | 50% |

These ablations suggest:

1. Relevant visual facts alone improve performance from 22.2% → 30.4%.
2. Removing the image but keeping relevant visual facts gives similar accuracy, 31.1%, suggesting that extracting useful diagram facts is a major bottleneck.
3. Adding relevant theorems improves accuracy to 39.1%.
4. Adding object-theorem bindings improves accuracy dramatically to 69.1%.
5. Removing visual facts while keeping theorem bindings drops performance to 50.0%, showing that theorem binding and visual grounding are both important.

To verify that theorem improvements were not merely due to longer prompts, I injected random theorems. Accuracy dropped back to 31.7%, close to the visual-facts-only setting.

### Main Diagnosis
The largest bottleneck is object-theorem binding.

The model may know a theorem such as the Pythagorean theorem, angle bisector theorem, or geometric mean theorem, but it often fails to bind the theorem to the correct sides, angles, or auxiliary points in the diagram.

The next major bottleneck is extracting useful visual facts from the diagram.








### 3. Theorem misapplication (or blindness)
example question: In triangle PQR, PS=8, QS=14. Find RS.


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

gen_prompt_correct stay between 95% and 100% over the training. gen_overall_correct increased from 40% range to 60% range.
| Reward mean | Reward std |
| :---: | :---: |
| ![](./assets/stage_1_mean_reward.png) | ![](./assets/stage_1_reward_std.png) | 

| KL| Entropy | Loss |
| :---: | :---: |:---: |
| ![](./assets/stage_1_kl.png) | ![](./assets/stage_1_entropy.png) |![](./assets/stage_1_loss.png) |

Stage 2

gen_prompt_correct stay between 80% and 90%. gen_overall_correct increased from 29-33% to 35-40% range
| Reward mean | Reward std |
| :---: | :---: |
| ![](./assets/stage_2_mean_reward.png) | ![](./assets/stage_2_reward_std.png) | 

| KL| Entropy | Loss |
| :---: | :---: |:---: |
| ![](./assets/stage_2_kl.png) | ![](./assets/stage_2_entropy.png) |![](./assets/stage_2_loss.png) |

Stage 3A

gen_prompt_correct and gen_overall_correct do not show clear improvement
| Reward mean | Reward std |
| :---: | :---: |
| ![](./assets/stage_3a_mean_reward.png) | ![](./assets/stage_3a_reward_std.png) | 

| KL| Entropy | Loss |
| :---: | :---: |:---: |
| ![](./assets/stage_3a_kl.png) | ![](./assets/stage_3a_entropy.png) |![](./assets/stage_3a_loss.png) |

Stage 3B

gen_prompt_correct and gen_overall_correct do not show clear improvement. The training dynamics does not show clear learning. 
| Reward mean | Reward std |
| :---: | :---: |
| ![](./assets/stage_3b_mean_reward.png) | ![](./assets/stage_3b_reward_std.png) | 

| KL| Entropy | Loss |
| :---: | :---: |:---: |
| ![](./assets/stage_3b_kl.png) | ![](./assets/stage_3b_entropy.png) |![](./assets/stage_3b_loss.png) |

### Evaluation at each stage
| Stage | Accuracy on Validation data |
|---|---|
| Stage 1 | 23%|
| Stage 2| 26.5%|
| Stage 3A|27.2%|
| Stage 3B|27.2%|

The improved accuracy from Stage 2 to Stage 3A is likely noise, so I picked the best checkpoint from Stage 3A. The evaluation accuracy on the test data is 28.9% compared to the baseline model's 22.4%.

### GRPO Analysis
I sampled 500 problems from stage 3A training data. The model's pass@8 on the sample is 65%, and its pass@16 is around 82%. I ran two versions of stage 3A one using k=8 and the other using k=16. However, the accuracy of the k=16 run is not higher than the k=8 run. Although the model clearly has some latent ability and can find correct solutions in its sampling tail, but GRPO is not effectively moving those solutions into the default high-probability behavior. I compared the problem types in the validation data the k=8 checkpoint and k=16 checkpoints solved respectively, and found that k=16 improved some problems types but regressed on other types. That looks like strategy shifting, not uniform improvement. The k=16 checkpoint may have become better at certain theorem families while forgetting or destabilizing others. So the overall average barely moves. The training reinforces whatever wins in the sampled rollouts, but it does not necessarily preserve broad geometry competence.

Stage 3B have more difficult problems than 3A, so it's expected that GRPO in stage 3B does not show improvement. Based on the findings from SFT and GRPO, I think a better way is to experiment on-policy ditillation: leverage the teacher model's ability in RL environment. 



## Stage 4 On-Policy Distillation (In Progress)


