# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import os
import re
import subprocess

import pytest
import numpy as np
from pathlib import Path

import import_helper

gpt2_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_gpt2_cmdline():
    cmd = ["python3", os.path.join(
        gpt2_root_dir, "text_generate_gpt2.py")]
    cmd.extend(["--model-name-or-path", "gpt2"])
    cmd.extend(["--fp16", "true"])
    cmd.extend(["--prompt", "My name is"])
    cmd.extend(["--single-ipu", "true"])
    cmd.extend(["--poptorch-loop", "true"])
    cmd.extend(["--output-len", "256"])

    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=gpt2_root_dir)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out.stdout.decode("utf-8"), out.stderr.decode("utf-8")


def parse_generation_results(out):
    latencys = []
    throughputs = []
    for line in out.split("\n"):
        print(line)
        match_latency = re.match(r".*(Latency: ([\d.]+)\ssec/token).*", line)
        match_throughput = re.match(
            r".*(Throughput: ([\d.]+)\ssentence).*", line)
        if match_latency:
            latency = match_latency.groups()[1]
            latencys.append(float(latency))
        if match_throughput:
            throughput = match_throughput.groups()[1]
            throughputs.append(float(throughput))
    latencys = np.array(latencys)
    throughputs = np.array(throughputs)

    return latencys, throughputs


def latency_reached_threshold(latencys):
    assert latencys.mean() < 0.002


def throughput_reached_threshold(throughouts):
    assert throughouts.mean() > 2.5


@pytest.mark.ipus(1)
def test_latency_throughput():
    """
    Test the GPT2 text generation.
    Latency and throughput should reach thresholds.
    """
    stdout, stderr = run_gpt2_cmdline()
    latencys, throughputs = parse_generation_results(stderr)
    print("latencys: ", latencys)
    print("throughputs: ", throughputs)
    latency_reached_threshold(latencys)
    throughput_reached_threshold(throughputs)
