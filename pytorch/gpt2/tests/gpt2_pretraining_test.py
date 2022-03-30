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

gpt2_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_gpt2_cmdline(script_path):
    try:
        out = subprocess.run(["bash", script_path], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=gpt2_root_dir)
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out.stdout.decode("utf-8"), out.stderr.decode("utf-8")


def parse_result_for_loss_accuracy_throughput(out):
    losses = []
    accs = []
    throughputs = []
    for line in out.split("\n"):
        match_loss = re.match(r".*(loss: ([\d.]+),).*", line)
        match_acc = re.match(r".*(acc: ([\d.]+),).*", line)
        match_throughput = re.match(r".*(Throughput: ([\d.]+)\s).*", line)
        if match_loss:
            loss = match_loss.groups()[1]
            losses.append(float(loss))
        if match_acc:
            acc = match_acc.groups()[1]
            accs.append(float(acc))
        if match_throughput:
            throughput = match_throughput.groups()[1]
            throughputs.append(float(throughput))
    losses = np.array(losses)
    accs = np.array(accs)
    throughputs = np.array(throughputs)
    return losses, accs, throughputs


def loss_not_none(losses):
    # Test that loss at end is less than loss at start
    assert len(losses) > 0


def accuracy_not_none(accs):
    # Test that accuracy at end is greater than accuracy at start
    assert len(accs) > 0


def throughput_reached_threshold(throughouts):
    assert throughouts.mean() > 180.0


@pytest.mark.ipus(2)
@pytest.mark.parametrize("script_path", [gpt2_root_dir + "/run/pretraining_test.sh"])
def test_loss_accuracy_throughput(script_path):
    """
    Test the GPT2 pretrining loop using generated dataset.
    Loss and accuracy should be not None and throughput should reach a threshold.
    """
    stdout, stderr = run_gpt2_cmdline(script_path)
    losses, accs, throughputs = parse_result_for_loss_accuracy_throughput(
        stderr)
    loss_not_none(losses)
    accuracy_not_none(accs)
    throughput_reached_threshold(throughputs)
