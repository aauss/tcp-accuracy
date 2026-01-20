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
"""TODO: Add a description here."""

import re

import evaluate
import datasets


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the module here
_DESCRIPTION = """\
This new module is designed to solve this great ML task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_module = evaluate.load("my_new_module")
    >>> results = my_new_module.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TCPAccuracy(evaluate.Metric):
    """TODO: Short description of my evaluation module."""

    BOXED_ANSWER_PATTERN = r"\\boxed\{([^}]*)\}"

    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            # Homepage of the module for documentation
            homepage="http://module.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_module"],
            reference_urls=["http://path.to.reference.url/new_module"],
        )

    def extract_boxed_answer(self, predictions):
        match = re.search(self.BOXED_ANSWER_PATTERN, predictions, re.DOTALL)
        if match:
            return match.group(1).replace("GMT", "").strip()
        return None

    def _compute(self, predictions, references, subset: str | list[str]):
        """Returns the scores"""
        # TODO: Compute the different scores of the module
        if isinstance(subset, str):
            subset = [subset] * len(predictions)
        predictions = [self.extract_boxed_answer(p) for p in predictions]
        references = [
            r.replace("GMT", "").strip() if s == "tcp_short" else r
            for r, s in zip(references, subset)
        ]
        accuracy = [int(i == j) for i, j in zip(predictions, references)]
        return {
            "accuracy": accuracy,
        }
