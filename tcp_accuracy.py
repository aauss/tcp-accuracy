# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TCP Accuracy metric for evaluating temporal constraint-based planning tasks."""

import re

import evaluate
import datasets


_CITATION = """\
@software{abbood2025tcp_accuracy,
  title={TCP Accuracy},
  author={Abbood, Auss},
  year={2025},
  url={https://huggingface.co/spaces/aauss/tcp_accuracy}
}
"""

_DESCRIPTION = """\
This metric evaluates model predictions on the TCP (Temporal Constraint-Based Planning) benchmark
(Ding et al., 2025). It measures accuracy by extracting answers from LaTeX boxed notation
(\\boxed{answer}) and comparing them against reference answers using exact string matching.
"""


_KWARGS_DESCRIPTION = """
Calculates accuracy for TCP benchmark predictions.
Args:
    predictions: list of prediction strings. Each prediction should contain the
        final answer in LaTeX boxed notation: \\boxed{answer}.
    references: list of reference answer strings.
    subset: either a string or list of strings indicating the subset type
        ("tcp_short" or "tcp_long"). For "tcp_short", GMT is stripped before comparison.
    return_average: if True (default), returns average accuracy as a float.
        If False, returns a list of binary scores (0 or 1) for each sample.
Returns:
    accuracy: float (if return_average=True) or list of int (if return_average=False)
Examples:
    >>> metric = evaluate.load("aauss/tcp_accuracy")
    >>> predictions = ["...\\\\boxed{2012-11-05}", "...\\\\boxed{2020-05-28 16:00}"]
    >>> references = ["2012-11-05", "2020-05-28 16:00 GMT"]
    >>> results = metric.compute(predictions=predictions, references=references, subset=["tcp_long", "tcp_short"])
    >>> print(results)
    {'accuracy': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TCPAccuracy(evaluate.Metric):
    """Accuracy metric for the TCP (Temporal Constraint-Based Planning) benchmark."""

    BOXED_ANSWER_PATTERN = r"\\boxed\{([^}]*)\}"

    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            homepage="https://huggingface.co/spaces/aauss/tcp_accuracy",
            codebase_urls=["https://huggingface.co/spaces/aauss/tcp_accuracy/tree/main"],
            reference_urls=["https://aclanthology.org/2025.emnlp-main.1142/"],
        )

    def extract_boxed_answer(self, prediction: str) -> str | None:
        match = re.search(self.BOXED_ANSWER_PATTERN, prediction, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _compute(
        self,
        predictions: list[str],
        references: list[str],
        subset: str | list[str],
        return_average: bool = True,
    ) -> dict[str, float | list[int]]:
        """Returns the scores"""
        if not predictions:
            raise ValueError("predictions cannot be empty")
        if isinstance(subset, str):
            subset = [subset] * len(predictions)
        extracted_predictions = [self.extract_boxed_answer(p) for p in predictions]
        extracted_predictions = [
            p.replace("GMT", "").strip() if p and s == "tcp_short" else p
            for p, s in zip(extracted_predictions, subset)
        ]
        references = [
            r.replace("GMT", "").strip() if s == "tcp_short" else r
            for r, s in zip(references, subset)
        ]
        accuracy = [int(i == j) for i, j in zip(extracted_predictions, references)]
        if return_average:
            return {"accuracy": sum(accuracy) / len(accuracy)}
        return {"accuracy": accuracy}
