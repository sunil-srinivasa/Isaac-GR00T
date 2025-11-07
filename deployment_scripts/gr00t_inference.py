# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import csv
import time

import torch
from action_head_utils import action_head_pytorch_forward
from trt_model_forward import setup_tensorrt_engines

import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def run_inference(policy, dataset, mode, num_steps=100, capture_nsys=False, log_to_csv=False):
    """Run inference on multiple samples and log timing stats."""
    N = min(len(dataset), num_steps)
    inference_times = torch.zeros(N, dtype=torch.float32)

    print(f"\nRunning {mode} inference on {N} samples...\n")

    if capture_nsys:
        nsys_start = N // 2
        num_nsys_steps = 5
        nsys_end = nsys_start + num_nsys_steps
    #     nsys_prefix = "nsys profile -s none -t cuda,nvtx -f true -c cudaProfilerApi --capture-range-end=\"repeat[]\" \
    #     --cuda-graph-trace=node --cuda-event-trace=false --cpuctxsw=none --cuda-flush-interval=10000 --gpu-metrics-frequency 50000"

    for i in range(N):
        step_data = dataset[i]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        # Wrap the inference call with NVTX if profiling is enabled
        if capture_nsys and nsys_start <= i < nsys_end:
            print("Capturing nsys...")
            with torch.cuda.nvtx.range("policy_inference"):
                predicted_action = policy.get_action(step_data)
        else:
            predicted_action = policy.get_action(step_data)        

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()

        inference_times[i] = end - start

        print(f"Sample {i+1}/{N} | Inference time: {end - start:.4f}s")

        if i == 0:
            print(f"\n=== {mode.capitalize()} Inference Results (Sample 1) ===")
            for key, value in predicted_action.items():
                print(f"{key}: {value.shape}")

    # Compute timing stats
    avg_time = torch.mean(inference_times).item()
    std_time = torch.std(inference_times).item()
    throughput = 1.0 / avg_time if avg_time > 0 else 0.0

    print(f"\n--- Summary ({mode}) ---")
    print(f"Average inference time: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")

    if log_to_csv:
        filename = f"inference_times_{mode}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_index", "inference_time_s"])
            for i, t in enumerate(inference_times):
                writer.writerow([i, float(t)])
        print(f"Saved timing log to {filename}")

    return inference_times


def compare_predictions(pred_tensorrt, pred_torch):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len("Cosine Similarity (PyTorch/TensorRT):"),
        len("L1 Mean/Max Distance (PyTorch/TensorRT):"),
        len("Max Output Values (PyTorch/TensorRT):"),
        len("Mean Output Values (PyTorch/TensorRT):"),
        len("Min Output Values (PyTorch/TensorRT):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(f"\n{key}:")
        print(f'{"Cosine Similarity (PyTorch/TensorRT):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{"L1 Mean/Max Distance (PyTorch/TensorRT):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{"Max Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{"Mean Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{"Min Output Values (PyTorch/TensorRT):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GR00T inference")
    parser.add_argument(
        "--model_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to the GR00T model"
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        choices=["pytorch", "tensorrt", "compare"],
        default="pytorch",
        help="Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference, 'compare' for compare PyTorch and TensorRT outputs similarity",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--trt_engine_path",
        type=str,
        help="Path to the TensorRT engine",
        default="gr00t_engine",
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
    DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
    EMBODIMENT_TAG = "gr1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    modality_config = policy.modality_config
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=EMBODIMENT_TAG,
    )

    step_data = dataset[0]

    if args.inference_mode == "pytorch":
        predicted_action = policy.get_action(step_data)
        print("\n=== PyTorch Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)
        run_inference(policy, dataset, "pytorch", capture_nsys=True)

    elif args.inference_mode == "tensorrt":
        # Setup TensorRT engines
        setup_tensorrt_engines(policy, args.trt_engine_path)

        predicted_action = policy.get_action(step_data)
        print("\n=== TensorRT Inference Results ===")
        for key, value in predicted_action.items():
            print(key, value.shape)

        run_inference(policy, dataset, "tensorrt", capture_nsys=True)

    else:
        # ensure PyTorch and TensorRT have the same init_actions
        if not hasattr(policy.model.action_head, "init_actions"):
            policy.model.action_head.init_actions = torch.randn(
                (1, policy.model.action_head.action_horizon, policy.model.action_head.action_dim),
                dtype=torch.float16,
                device=device,
            )
        # PyTorch inference
        policy.model.action_head.get_action = partial(
            action_head_pytorch_forward, policy.model.action_head
        )
        predicted_action_torch = policy.get_action(step_data)

        # Setup TensorRT engines and run inference
        setup_tensorrt_engines(policy, args.trt_engine_path)
        predicted_action_tensorrt = policy.get_action(step_data)

        # Compare predictions
        compare_predictions(predicted_action_tensorrt, predicted_action_torch)
