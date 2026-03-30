# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Blendable dataset."""

import hashlib
import os
import time

import ezpz

import torch
import numpy as np

import blendcorpus.parallel_state as mpu
from blendcorpus.utils import Profile, PerfTrace, get_logger

logger = ezpz.get_logger(__name__)

dlp = Profile("DATASET")


class BlendableDataset(torch.utils.data.Dataset):
    @dlp.log
    def __init__(self, datasets, weights, size, *, data_cache_path=None):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indicies.
        @dlp.log
        def _build_indices():
            start_time = time.perf_counter()
            dataset_index = np.zeros(self.size, dtype=np.int64)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from blendcorpus.data import helpers

            helpers.build_blending_indices(
                dataset_index,
                dataset_sample_index,
                weights,
                num_datasets,
                self.size,
                torch.distributed.get_rank() == 0,
            )
            logger.info(
                "> elapsed time for building blendable dataset indices: "
                f"{time.perf_counter() - start_time:.2f} (sec)"
            )
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc
        self.dataset_index = np.zeros(self.size, dtype=np.int64)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        # Build blending indices on rank 0 and broadcast to all ranks.
        # Previous approach used filesystem caching, but Lustre metadata
        # propagation delays cause FileNotFoundError on multi-node runs.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            logger.info("> building blendable dataset indices on rank 0 ...")
            self.dataset_index, self.dataset_sample_index = _build_indices()
            # Also save to cache for future single-rank reuse
            if data_cache_path:
                try:
                    stable_desc = f"Blendable dataset\nWeights: {weights}\nSize: {size}\n"
                    desc_hash = hashlib.md5(stable_desc.encode("utf-8")).hexdigest()
                    os.makedirs(data_cache_path, exist_ok=True)
                    np.save(os.path.join(data_cache_path, desc_hash + "_index.npy"),
                            self.dataset_index, allow_pickle=True)
                    np.save(os.path.join(data_cache_path, desc_hash + "_sample_index.npy"),
                            self.dataset_sample_index, allow_pickle=True)
                    logger.info(f"> saved blendable index cache to {data_cache_path}")
                except OSError:
                    logger.warning(f"> failed to save blendable index cache to {data_cache_path}")

        # Broadcast indices from rank 0 to all ranks via torch.distributed
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            idx_tensor = torch.from_numpy(self.dataset_index).long()
            sample_tensor = torch.from_numpy(self.dataset_sample_index).long()
            torch.distributed.broadcast(idx_tensor, src=0)
            torch.distributed.broadcast(sample_tensor, src=0)
            if rank != 0:
                self.dataset_index = idx_tensor.numpy()
                self.dataset_sample_index = sample_tensor.numpy()
            logger.info(f"> rank {rank}: blendable indices ready (broadcast)")
        elif not torch.distributed.is_initialized():
            self.dataset_index, self.dataset_sample_index = _build_indices()

        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError("BlendedDataset size is improperly bounded")
        except IndexError:
            pass
        logger.info("> size of blendable dataset: {} samples".format(self.size))

    def __len__(self):
        return self.size

    @dlp.log
    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx": dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }
