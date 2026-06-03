# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import types
from unittest.mock import Mock, patch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats


class _Request:

    def __init__(self, *, context_chunk_size=0, last_context_chunk=None):
        self.context_chunk_size = context_chunk_size
        self.py_last_context_chunk = last_context_chunk


class _ScheduledBatch:

    def __init__(self, *, context_requests=None, generation_requests=None):
        self.context_requests = context_requests or []
        self.generation_requests = generation_requests or []
        self.paused_requests = []

    @property
    def num_context_requests(self):
        return len(self.context_requests)

    @property
    def num_generation_requests(self):
        return len(self.generation_requests)


def _fake_executor(iter_states):
    fake = types.SimpleNamespace()
    fake.executor_request_queue = Mock()
    fake.executor_request_queue.get_request_queue_size.return_value = 0
    fake.max_num_active_requests = 0
    fake.resource_manager = types.SimpleNamespace(resource_managers={})
    fake.iter_counter = 1
    fake.model_engine = types.SimpleNamespace(iter_states=iter_states)
    return fake


def _update_iter_stats(fake_executor, scheduled_batch):
    stats = IterationStats()
    stats.inflight_batching_stats = InflightBatchingStats()

    with patch(
            "tensorrt_llm._torch.pyexecutor.py_executor.torch.cuda.mem_get_info",
            return_value=(1 << 30, 1 << 30)):
        return PyExecutor._update_iter_stats(fake_executor,
                                             stats,
                                             iter_latency_ms=1.0,
                                             num_completed_requests=0,
                                             scheduled_batch=scheduled_batch,
                                             micro_batch_id=0)


def test_num_ctx_tokens_uses_scheduled_batch_snapshot_not_iter_states():
    fake = _fake_executor({
        "num_ctx_requests": 0,
        "num_ctx_tokens": 99999,
        "num_generation_tokens": 1,
    })
    scheduled_batch = _ScheduledBatch(context_requests=[
        _Request(last_context_chunk=(0, 128)),
        _Request(last_context_chunk=(256, 512)),
    ])

    stats = _update_iter_stats(fake, scheduled_batch)

    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 2
    assert ifb.num_ctx_tokens == 384
    assert fake.model_engine.iter_states["num_ctx_requests"] == 2
    assert fake.model_engine.iter_states["num_ctx_tokens"] == 384
    assert fake.model_engine.iter_states["num_generation_tokens"] == 1


def test_num_ctx_tokens_zero_for_decode_only_with_stale_iter_states():
    fake = _fake_executor({
        "num_ctx_requests": 1,
        "num_ctx_tokens": 2048,
        "num_generation_tokens": 1,
    })
    scheduled_batch = _ScheduledBatch(generation_requests=[_Request()])

    stats = _update_iter_stats(fake, scheduled_batch)

    ifb = stats.inflight_batching_stats
    assert ifb.num_context_requests == 0
    assert ifb.num_gen_requests == 1
    assert ifb.num_ctx_tokens == 0
    assert fake.model_engine.iter_states["num_ctx_requests"] == 0
    assert fake.model_engine.iter_states["num_ctx_tokens"] == 0
