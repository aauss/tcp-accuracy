---
title: TCP Accuracy
datasets:
- Beanbagdzf/TCP
tags:
- evaluate
- metric
- temporal reasoning
- scheduling
- temporal constraint programming
description: >-
  Accuracy metric for the TCP (Temporal Constraint-Based Planning) benchmark by
  Ding et al. (2025).
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
emoji: ⏰
colorFrom: pink
colorTo: red
---

# Metric Card for TCP Accuracy

## Metric Description

This metric is designed for the **TCP** (Temporal Constraint-Based Planning) benchmark (Ding et al., 2025). It evaluates large language models on complex scheduling and planning tasks that require temporal reasoning, constraint satisfaction, and multi-step logical deduction.

The benchmark includes problems such as:
- Project scheduling with team member availability constraints
- Work duration limits and mandatory break requirements
- Time zone conversions
- Sequential task dependencies

The metric expects model outputs to contain the final answer in LaTeX boxed notation: `\boxed{answer}`.

It performs the following steps:

1. Extracts the answer from the model's prediction string using the regex pattern `\boxed{([^}]*)}`.
2. For the "tcp_short" subset, removes "GMT" from both predictions and references before comparison.
3. Performs exact string matching between the extracted prediction and the reference answer.
4. Returns accuracy as the proportion of correct matches (or per-sample scores).

## How to Use

You can load the metric using the `evaluate` library:

```python
import evaluate

metric = evaluate.load("aauss/tcp_accuracy")

predictions = [
    "After analyzing the constraints... \\boxed{2012-11-05}",
    "The project completes on... \\boxed{2021-01-10}",
    "Converting to GMT, the final time is... \\boxed{2020-05-28 16:00}",
]

references = ["2012-11-05", "2012-11-05", "2020-05-28 16:00 GMT"]
subsets = ["tcp_long", "tcp_long", "tcp_short"]

# Get average accuracy
result = metric.compute(
    predictions=predictions,
    references=references,
    subset=subsets,
)
print(result)
>>> {"accuracy": 0.6666666666666666}

# Get per-sample accuracy
result = metric.compute(
    predictions=predictions,
    references=references,
    subset=subsets,
    return_average=False,
)
print(result)
>>> {"accuracy": [1, 0, 1]}
```

### Inputs

- **predictions** (`list` of `str`): List of predictions to score. Each prediction should be a string containing the model's response, which must include the final answer in the format `\boxed{answer}`.
- **references** (`list` of `str`): List of reference answers. Each reference should be the expected answer string.
- **subset** (`str` or `list` of `str`): The subset type(s) for each sample. Must be one of:
  - `"tcp_long"`: Longer scheduling problems (exact match)
  - `"tcp_short"`: Shorter problems (GMT is stripped before comparison)
- **return_average** (`bool`, optional): If `True`, returns the average accuracy as a float. If `False`, returns a list of binary scores (1 for correct, 0 for incorrect) for each sample. Defaults to `True`.

### Output Values

The metric returns a dictionary with the following key:

- **accuracy** (`float` or `list` of `int`): The accuracy score (0.0 to 1.0) if `return_average=True`, or a list of binary values (0 or 1) indicating correctness per sample if `return_average=False`.

This metric can take on any value between 0.0 and 1.0, inclusive. Higher scores indicate better performance.

#### Values from Popular Papers

Refer to the [original TCP paper](https://aclanthology.org/2025.emnlp-main.1142/) for baseline performance values across various language models.

## Limitations and Bias

- The metric relies on the regex pattern `\boxed{([^}]*)}` to extract answers. If the model output does not include a boxed answer, extraction will fail and return `None`, resulting in an incorrect prediction.
- For "tcp_short" subset, "GMT" is stripped from both predictions and references. Other timezone formats may not be handled correctly.
- The metric uses exact string matching. Semantically equivalent answers with different formatting (e.g., "Nov 5, 2012" vs "2012-11-05") will be marked as incorrect.
- Nested braces inside `\boxed{}` are not supported by the current regex pattern.

## Citation

```bibtex
@software{abbood2025tcp_accuracy,
  title={TCP Accuracy},
  author={Abbood, Auss},
  year={2025},
  url={https://huggingface.co/spaces/aauss/tcp_accuracy}
}
```

## Further References

- [TCP Paper (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1142/)
- [TCP Dataset on Hugging Face](https://huggingface.co/datasets/Beanbagdzf/TCP)
