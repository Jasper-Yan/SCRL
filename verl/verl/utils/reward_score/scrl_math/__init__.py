# Copyright 2026 SCRL Team (https://arxiv.org/abs/2603.19880)
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

"""
SCRL reward function for math tasks.
Uses pseudo-labeling with confidence thresholds and entropy-based negative sampling.
"""

from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade, compute_score
import traceback


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    Reward function for SCRL.
    
    Two reward modes:
    1. Binary mode:
       r_i = +a_plus if a_i = y^+
       r_i = -a_minus if a_i in N^-
       r_i = 0 otherwise
    
    2. Continuous mode:
       r_i = a_plus * (p_ai - tau_pos) - lambda_H * (H_ai - H_bar) if a_i = y^+
       r_i = -a_minus * (tau_low - p_ai) - lambda_H * (H_ai - H_bar) if a_i in N^-
       r_i = 0 - lambda_H * (H_ai - H_bar) otherwise
    
    Args:
        data_source: Source of the data
        solution_str: Model's solution string
        ground_truth: Pseudo-positive label (could be None)
        extra_info: Dict with keys:
            - reward_mode: "binary" or "continuous"
            - negative_set: List of negative answers
            - a_plus, a_minus: Reward scaling factors
            - lambda_h: Entropy penalty weight (for continuous mode)
            - answer_proportions: Dict[str, float] (for continuous mode)
            - intra_class_entropys: Dict[str, float] (for continuous mode)
            - overall_avg_entropy: float (for continuous mode)
            - tau_pos, tau_low: Thresholds (for continuous mode)
            - trajectory_avg_entropy: float (for continuous mode)
        sandbox_fusion_url: URL for sandbox fusion
        concurrent_semaphore: Semaphore for concurrent execution
    
    Returns:
        Reward score (float or dict)
    """
    try:
        # Extract model's answer
        model_answer = extract_answer(solution_str)
        
        if model_answer is None:
            return {
                "score": -1.0,
                "format_score": 0.0,
                "acc": False,
                "extracted_gt": str(ground_truth) if ground_truth is not None else "",
                "pred": "",
                "is_negative": False,
            }
        
        # Get parameters from extra_info
        reward_mode = "binary"
        a_plus = 1.0
        a_minus = 1.0
        negative_set = []
        lambda_h = 0.0
        
        if extra_info is not None and isinstance(extra_info, dict):
            reward_mode = extra_info.get("reward_mode", "binary")
            a_plus = extra_info.get("a_plus", 1.0)
            a_minus = extra_info.get("a_minus", 1.0)
            negative_set = extra_info.get("negative_set", [])
            lambda_h = extra_info.get("lambda_h", 0.0)
        
        # Determine which case this answer belongs to
        is_pseudo_positive = False
        is_negative = False
        
        # Check if matches pseudo-positive label
        if ground_truth is not None:
            is_pseudo_positive = grade(model_answer, str(ground_truth), fast=False)
        
        # Check if in negative set
        if not is_pseudo_positive and negative_set:
            for negative_answer in negative_set:
                if grade(model_answer, negative_answer, fast=False):
                    is_negative = True
                    break
        
        # Compute reward based on mode
        if reward_mode == "binary":
            # Binary mode: simple +1, -1, 0
            if is_pseudo_positive:
                reward = 1.0
                acc = True
            elif is_negative:
                reward = -1.0
                acc = False
            else:
                reward = 0.0
                acc = False
        
        else:  # continuous mode
            # Get continuous mode parameters
            answer_proportions = extra_info.get("answer_proportions", {})
            intra_class_entropys = extra_info.get("intra_class_entropys", {})
            overall_avg_entropy = extra_info.get("overall_avg_entropy", 0.0)
            tau_pos = extra_info.get("tau_pos", 0.5)
            tau_low = extra_info.get("tau_low", 0.15)
            trajectory_avg_entropy = extra_info.get("trajectory_avg_entropy", 0.0)
            
            # Get answer proportion and intra-class entropy for this answer
            model_answer_simplified = simplify_expression_string(model_answer)
            p_ai = answer_proportions.get(model_answer_simplified, 0.0)
            H_ai = intra_class_entropys.get(model_answer_simplified, trajectory_avg_entropy)
            H_bar = overall_avg_entropy
            """
            # DEBUG: Print parameter values for first few calls
            print(f"  model_answer: {model_answer}")
            print(f"  model_answer_simplified: {model_answer_simplified}")
            print(f"  is_pseudo_positive: {is_pseudo_positive}, is_negative: {is_negative}")
            print(f"  extra_info keys: {list(extra_info.keys()) if extra_info else 'None'}")
            print(f"  answer_proportions: {answer_proportions}")
            print(f"  intra_class_entropys: {intra_class_entropys}")
            print(f"  p_ai (from dict): {p_ai}")
            print(f"  H_ai (from dict): {H_ai}")
            print(f"  H_bar (overall): {H_bar}")
            print(f"  trajectory_avg_entropy (fallback): {trajectory_avg_entropy}")
            print(f"  tau_pos: {tau_pos}, tau_low: {tau_low}")
            print(f"  a_plus: {a_plus}, a_minus: {a_minus}, lambda_h: {lambda_h}")
            """
            # Entropy penalty term
            entropy_penalty = lambda_h * (H_ai - H_bar)
            
            if is_pseudo_positive:
                # r_i = a_plus * (p_ai - tau_pos) - lambda_H * (H_ai - H_bar)
                reward = a_plus * (p_ai) - entropy_penalty
                acc = True
            elif is_negative:
                # r_i = -a_minus * (tau_low - p_ai) - lambda_H * (H_ai - H_bar)
                reward = -a_minus * (tau_low - p_ai) - entropy_penalty
                acc = False
            else:
                # r_i = 0 - lambda_H * (H_ai - H_bar)
                reward = -entropy_penalty
                acc = False
        
        return {
            "score": reward,
            "format_score": 1.0,
            "acc": acc,
            "extracted_gt": str(ground_truth) if ground_truth is not None else "",
            "pred": model_answer,
            "is_negative": is_negative,
        }
        
    except Exception as e:
        print(f"[ERROR] Error in scrl reward_func: {str(e)}")
        traceback.print_exc()
        raise
