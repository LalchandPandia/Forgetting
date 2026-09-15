"""Shared MedCalc-Bench answer parsing and grading logic.

Used by both eval_medcalc.py (transformers/OLMo) and eval_medcalc_qwen.py
(vLLM/Qwen) so the two evaluators can't drift onto different notions of
"correct" for the same benchmark.

Grading follows the benchmark's own protocol (Khandekar et al. 2024,
arXiv:2406.12036, Sec 3.1):
  - date calculators                      -> exact match
  - rule-based (risk/severity/diagnosis)  -> exact match
  - equation-based lab/physical/dosage    -> within TOLERANCE relative error,
                                              or the dataset's own
                                              [lower_limit, upper_limit] band
                                              when the row provides one
"""

import json
import re

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUM_RE = re.compile(r"-?\d+\.?\d*")

RULE_BASED_CATEGORIES = {"risk", "severity", "diagnosis"}
TOLERANCE = 0.05


def extract_answer(completion):
    """Return the answer string from a model completion, or None if unparseable.

    Tries, in order: the strict {"answer": ...} JSON the prompt asks for; a
    forgiving regex for a near-JSON blob (single quotes, trailing text); and,
    as a last resort, the raw <answer> block itself if it's short enough to
    plausibly be a bare answer.
    """

    match = ANSWER_RE.search(completion)
    if not match:
        return None
    blob = match.group(1).strip()

    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"]).strip()
    except json.JSONDecodeError:
        pass

    fallback_match = re.search(r'["\']answer["\']\s*:\s*["\']?([^"\'}\n]+)', blob)
    if fallback_match:
        return fallback_match.group(1).strip()

    return blob if blob and len(blob) < 100 else None


def norm(s):
    return str(s).strip().strip('."').lower()


def to_float(s):
    if s is None:
        return None
    match = NUM_RE.search(str(s).replace(",", ""))
    return float(match.group()) if match else None


def grade(pred, ground_truth, category=None, output_type=None, lower_limit=None, upper_limit=None):
    """Type-dependent correctness, per the benchmark's three rules above."""

    if pred is None:
        return False

    category = str(category or "").lower()
    output_type = str(output_type or "").lower()

    # Dates: exact match.
    if "date" in category or "date" in output_type:
        return norm(pred) == norm(ground_truth)

    # Rule-based scores: exact match.
    if any(k in category for k in RULE_BASED_CATEGORIES):
        p, g = to_float(pred), to_float(ground_truth)
        if p is not None and g is not None:
            return p == g
        return norm(pred) == norm(ground_truth)

    # Equation-based lab / physical / dosage: within tolerance.
    p, g = to_float(pred), to_float(ground_truth)
    if p is None or g is None:
        return norm(pred) == norm(ground_truth)

    # Prefer the dataset's own explicit bounds when the row provides them.
    lo, hi = to_float(lower_limit), to_float(upper_limit)
    if lo is not None and hi is not None:
        return lo <= p <= hi

    if g == 0:
        return p == 0
    return abs(p - g) / abs(g) <= TOLERANCE
