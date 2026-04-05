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

from typing import List, Tuple, Optional, Dict
from collections import Counter
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade


def compute_trajectory_entropy(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute average entropy for each trajectory.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        attention_mask: (batch_size, seq_len)
    
    Returns:
        avg_entropy: (batch_size,) - average entropy per trajectory
    """
    # Compute token-level entropy: H_t = -sum(p * log(p))
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)
    
    # Mask out padding tokens
    token_entropy = token_entropy * attention_mask
    
    # Average over valid tokens
    valid_length = attention_mask.sum(dim=-1).clamp(min=1)
    avg_entropy = token_entropy.sum(dim=-1) / valid_length
    
    return avg_entropy


def determine_pseudo_labels(
    model_outputs: List[str],
    avg_entropys: np.ndarray,
    n: int,
    tau_pos: float = 0.5,
    tau_marg: float = 0.15,
    tau_low_min: float = 0.15,
) -> Tuple[List[Optional[str]], List[List[str]], List[Dict]]:
    """
    Determine pseudo-positive labels and negative sample sets for each prompt.
    
    Args:
        model_outputs: List of model output strings
        avg_entropys: Array of average entropys for each trajectory
        n: Number of samples per prompt
        tau_pos: Threshold for positive label proportion
        tau_marg: Margin threshold between top-2 answers
        tau_low_min: Minimum threshold for low proportion
    
    Returns:
        pseudo_positive_labels: List of pseudo-positive labels (None if not confident enough)
        negative_sets: List of negative answer sets for each prompt
        label_stats: List of statistics dictionaries for each prompt
    """
    assert len(model_outputs) % n == 0
    assert len(avg_entropys) == len(model_outputs)
    
    n_prompts = len(model_outputs) // n
    pseudo_positive_labels = []
    negative_sets = []
    label_stats_list = []
        
    for i in range(n_prompts):
        start_idx = i * n
        end_idx = start_idx + n
        
        prompt_outputs = model_outputs[start_idx:end_idx]
        prompt_entropys = avg_entropys[start_idx:end_idx]
        
        # Extract and simplify answers
        _NO_ANSWER = "__NO_ANSWER__"
        model_answers = [extract_answer(text) for text in prompt_outputs]
        model_answers = [
            simplify_expression_string(answer) if answer is not None else _NO_ANSWER
            for answer in model_answers
        ]
        
        # Count answer frequencies
        counter = Counter(model_answers)
        total_count = len(model_answers)
        
        # Sort by frequency
        sorted_answers = counter.most_common()
        
        # Compute proportions and intra-class entropys
        answer_props = {}
        intra_class_entropys = {}
        
        for answer, count in sorted_answers:
            answer_props[answer] = count / total_count
            # Get entropys for this answer class
            answer_indices = [j for j, a in enumerate(model_answers) if a == answer]
            intra_class_entropys[answer] = np.mean([prompt_entropys[j] for j in answer_indices])
        
        # Overall average entropy for this prompt
        overall_avg_entropy = np.mean(prompt_entropys)
        
        # Determine pseudo-positive label
        pseudo_positive = None
        # Filter out the sentinel "no answer" from candidates
        candidate_answers = [(a, c) for a, c in sorted_answers if a != _NO_ANSWER]
        if len(candidate_answers) >= 1:
            top_answer, top_count = candidate_answers[0]
            top_prop = answer_props[top_answer]
            
            # Check if top answer meets criteria
            if top_prop >= tau_pos:
                if len(candidate_answers) >= 2:
                    second_prop = answer_props[candidate_answers[1][0]]
                    margin = top_prop - second_prop
                    if margin > tau_marg:
                        pseudo_positive = top_answer
                else:
                    # Only one unique answer with high proportion
                    pseudo_positive = top_answer
        
        # Determine negative samples
        negative_answers = []
        tau_low = tau_low_min
        
        for answer, count in sorted_answers:
            prop = answer_props[answer]
            avg_h = intra_class_entropys[answer]
            
            # Only extremely low proportion AND less confident (higher entropy)
            if answer != pseudo_positive and answer != _NO_ANSWER and prop < tau_low_min and avg_h >= overall_avg_entropy:
                negative_answers.append(answer)
        
        pseudo_positive_labels.append(pseudo_positive)
        negative_sets.append(negative_answers)
        
        # Statistics
        label_stats = {
            "answer_distribution": dict(counter),
            "answer_proportions": answer_props,
            "intra_class_entropys": intra_class_entropys,
            "overall_avg_entropy": overall_avg_entropy,
            "pseudo_positive": pseudo_positive,
            "negative_set": negative_answers,
            "tau_low": tau_low,
        }
        label_stats_list.append(label_stats)
            
    return pseudo_positive_labels, negative_sets, label_stats_list


def apply_scrl_gt(batch, gen_batch_output, n, tokenizer, config=None):
    """
    Apply SCRL ground truth to the batch.
    Uses pseudo-labeling with confidence thresholds and entropy-based negative sampling.
    
    Args:
        batch: Training batch
        gen_batch_output: Generated rollouts
        n: Number of samples per prompt
        tokenizer: Tokenizer for decoding
        config: Configuration dict with thresholds (tau_pos, tau_marg, etc.)
    
    Returns:
        Updated batch with pseudo-labels and negative sets
    """
    
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"
    
    # Extract model outputs and compute entropys
    model_outputs = []
    avg_entropys_list = []
    
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)
            
            # Compute trajectory entropy from token_entropys if available
            if "token_entropys" in data_item.batch:
                token_entropys = data_item.batch["token_entropys"]
                prompt_length = prompt_ids.shape[-1]
                attention_mask = data_item.batch["attention_mask"]
                response_mask = attention_mask[prompt_length:]
                
                # Compute average entropy over valid (non-padding) response tokens
                valid_entropys = token_entropys * response_mask
                valid_token_count = response_mask.sum().clamp(min=1)
                avg_entropy = (valid_entropys.sum() / valid_token_count).item()
            else:
                # Fallback: compute from logits if available
                if "logits" in data_item.batch:
                    logits = data_item.batch["logits"]
                    attention_mask = data_item.batch["attention_mask"][prompt_length:]
                    avg_entropy = compute_trajectory_entropy(
                        logits.unsqueeze(0), attention_mask.unsqueeze(0)
                    )[0].item()
                else:
                    # Last fallback: use default value
                    avg_entropy = 0.0
            
            avg_entropys_list.append(avg_entropy)
            
            # Save trajectory-level entropy to gen_batch_output for later use in reward_func
            if "extra_info" not in data_item.non_tensor_batch:
                data_item.non_tensor_batch["extra_info"] = {}
            data_item.non_tensor_batch["extra_info"]["trajectory_avg_entropy"] = avg_entropy
    
    avg_entropys = np.array(avg_entropys_list)
    
    # Get configuration parameters
    tau_pos = config.get("tau_pos", 0.5) if config else 0.5
    tau_marg = config.get("tau_marg", 0.15) if config else 0.15
    tau_low_min = config.get("tau_low_min", 0.15) if config else 0.15
    a_plus = config.get("a_plus", 1.0) if config else 1.0
    a_minus = config.get("a_minus", 1.0) if config else 1.0
    reward_mode = config.get("reward_mode", "binary") if config else "binary"
    lambda_h = config.get("lambda_h", 0.1) if config else 0.1
    
    # Determine pseudo-labels and negative sets (with debug output inside)
    pseudo_positive_labels, negative_sets, label_stats_list = determine_pseudo_labels(
        model_outputs=model_outputs,
        avg_entropys=avg_entropys,
        n=n,
        tau_pos=tau_pos,
        tau_marg=tau_marg,
        tau_low_min=tau_low_min,
    )
    
    # Apply to batch and store reward parameters
    # Create extra_info array for batch if it doesn't exist
    if "extra_info" not in batch.non_tensor_batch:
        batch.non_tensor_batch["extra_info"] = np.array(
            [{} for _ in range(num_prompts)], dtype=object
        )
    
    batch_extra_info_array = batch.non_tensor_batch["extra_info"]
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        
        # Store pseudo-positive label (could be None)
        pseudo_positive = pseudo_positive_labels[i]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = pseudo_positive
        data_item.non_tensor_batch["reward_model"]["pseudo_positive"] = pseudo_positive
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        data_item.non_tensor_batch["reward_model"]["negative_set"] = negative_sets[i]
        data_item.non_tensor_batch["reward_model"]["label_stats"] = label_stats_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = original_gt  # Keep original GT as majority_gt for reference
        
        # Store reward parameters in batch's extra_info
        batch_extra_info_array[i]["reward_mode"] = reward_mode
        batch_extra_info_array[i]["negative_set"] = negative_sets[i]
        batch_extra_info_array[i]["a_plus"] = a_plus
        batch_extra_info_array[i]["a_minus"] = a_minus
        batch_extra_info_array[i]["lambda_h"] = lambda_h
        
        # For continuous mode, add additional parameters
        if reward_mode == "continuous":
            label_stats = label_stats_list[i]
            batch_extra_info_array[i]["answer_proportions"] = label_stats["answer_proportions"]
            batch_extra_info_array[i]["intra_class_entropys"] = label_stats["intra_class_entropys"]
            batch_extra_info_array[i]["overall_avg_entropy"] = label_stats["overall_avg_entropy"]
            batch_extra_info_array[i]["tau_pos"] = tau_pos
            batch_extra_info_array[i]["tau_low"] = tau_low_min
    
    pseudo_positive_ratio = sum([1 for label in pseudo_positive_labels if label is not None]) / num_prompts
    avg_num_negatives = np.mean([len(neg_set) for neg_set in negative_sets])
    
    batch.non_tensor_batch["pseudo_positive_ratio"] = np.full(num_prompts, pseudo_positive_ratio)
    batch.non_tensor_batch["avg_num_negatives"] = np.full(num_prompts, avg_num_negatives)
    
    return batch

def compute_scrl_metrics(
    batch,
    n: int,
):
    """
    Compute metrics for SCRL.
    
    Args:
        batch: Batch data
        n: Number of samples per prompt
        pseudo_positive_labels: Pseudo-positive labels
        negative_sets: Negative answer sets
        gt_labels: Ground truth labels
    
    Returns:
        metrics: Dictionary of metrics
    """
    assert len(batch) % n == 0
    num_prompts = len(batch) // n

    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    rewards = []  
    has_pseudo_positive = []
    num_negatives = []
    prompt_pass = []

    for p in range(num_prompts):
        prompt_indices = idx[p * n : (p + 1) * n]

        prompt_rewards = []
        prompt_rewards_original = []
        for bi in prompt_indices:
            data_item = batch[bi]
            prompt_rewards.append(data_item.batch["token_level_scores"].sum().item())
            if "token_level_scores_original" in data_item.batch:
                prompt_rewards_original.append(data_item.batch["token_level_scores_original"].sum().item())
        rewards.extend(prompt_rewards)

        first_item = batch[prompt_indices[0]]
        pseudo_pos = first_item.non_tensor_batch["reward_model"].get("pseudo_positive", None)
        neg_set = first_item.non_tensor_batch["reward_model"].get("negative_set", [])
        gt_answer = first_item.non_tensor_batch["reward_model"].get("original_gt", "")

        has_pseudo_positive.append(pseudo_pos is not None)
        num_negatives.append(len(neg_set))

        if prompt_rewards_original:
            prompt_pass.append(1.0 if sum(prompt_rewards_original) >= 1.0 else 0.0)
        else:
            prompt_pass.append(1.0 if sum(prompt_rewards) >= 1.0 else 0.0)

    label_hits = 0.0
    count_with_pseudo = 0
    for p in range(num_prompts):
        prompt_indices = idx[p * n : (p + 1) * n]
        gt_answer = batch[prompt_indices[0]].non_tensor_batch["reward_model"].get("original_gt", "")
        pseudo_pos = batch[prompt_indices[0]].non_tensor_batch["reward_model"].get("pseudo_positive", None)
        if pseudo_pos is not None:
            count_with_pseudo += 1
            if grade(pseudo_pos, gt_answer):
                label_hits += 1.0
    label_accuracy = label_hits / count_with_pseudo if count_with_pseudo > 0 else 0.0

    total_negative_labels = 0
    correct_negative_labels = 0
    for p in range(num_prompts):
        prompt_indices = idx[p * n : (p + 1) * n]
        gt_answer = batch[prompt_indices[0]].non_tensor_batch["reward_model"].get("original_gt", "")
        neg_set = batch[prompt_indices[0]].non_tensor_batch["reward_model"].get("negative_set", [])
        for neg_answer in neg_set:
            total_negative_labels += 1
            if not grade(neg_answer, gt_answer):
                correct_negative_labels += 1
    negative_label_accuracy = (
        correct_negative_labels / total_negative_labels if total_negative_labels > 0 else 0.0
    )

    metrics = {
        "scrl/postive_label_accuracy": label_accuracy,
        "scrl/avg_reward": np.mean(rewards) if rewards else 0.0,
        "scrl/pseudo_positive_ratio": sum(has_pseudo_positive) / num_prompts,
        "scrl/avg_num_negatives": np.mean(num_negatives) if num_negatives else 0.0,
        f"scrl/pass@{n}": sum(prompt_pass) / num_prompts,
        "scrl/negative_label_accuracy": negative_label_accuracy,
        "scrl/total_negative_labels": total_negative_labels,
    }

    return metrics