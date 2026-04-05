# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


def _to_scalar_score(res):
    """Normalize reward_fn outputs to a float score.
    """

    if isinstance(res, dict):
        return float(res.get("score", 0.0))
    if isinstance(res, (list, tuple)) and len(res) > 0:
        return float(res[0])
    return float(res)


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    # score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    # score_lst = [reward_fn(r, ground_truth) for r in response_lst]

    raw_scores = [reward_fn(r, ground_truth) for r in response_lst]
    score_lst = [_to_scalar_score(res) for res in raw_scores]

    # Return all scores for pass@k calculation
    return data_source, score_lst


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    # evaluate test_score based on data source
    data_source_scores = defaultdict(list)  # Store all scores for pass@k calculation
    reward_fn = get_custom_reward_fn(config)
    
    # If no custom reward function is specified, use the default one
    if reward_fn is None:
        from verl.utils.reward_score.ttrl_math import compute_score
        # from verl.utils.reward_score.math_verify import compute_score
        print("No custom reward function specified, using default_compute_score based on data_source")
        reward_fn = compute_score

    # Create remote tasks
    remote_tasks = [
        process_item.remote(reward_fn, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, scores = ray.get(result_id)
                data_source_scores[data_source].append(scores)
                pbar.update(1)

    metric_dict = {}
    for data_source, all_scores in data_source_scores.items():
        # all_scores is a list of lists, where each inner list contains scores for one problem
        
        # Calculate average score (original metric)
        avg_scores = [np.mean(scores) for scores in all_scores]
        metric_dict[f"test_score/{data_source}_avg"] = np.mean(avg_scores)
        
        # Calculate pass@1: average of individual scores (expected pass rate when sampling 1)
        all_individual_scores = [score for scores in all_scores for score in scores]
        metric_dict[f"test_score/{data_source}_pass@1"] = np.mean(all_individual_scores)

        # Calculate pass@16: percentage of problems where at least one solution is correct
        pass_at_16 = np.mean([np.max(scores) > 0.5 for scores in all_scores])  # assuming score > 0.5 means correct
        metric_dict[f"test_score/{data_source}_pass@16"] = pass_at_16

    print(metric_dict)


if __name__ == "__main__":
    main()
