"""Load Chronos workload definitions under Nimble's legacy PyTorch fork.

The main validation module imports torch.compile/torch.library infrastructure
that does not exist in PyTorch 1.7. This loader executes only its model and
input-wrapper section and supplies a legacy-compatible stateful LIF node.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet, spiking_vgg


class CustomStatefulIFNode(nn.Module):
    def __init__(
        self,
        v_threshold=1.0,
        v_reset=0.0,
        tau=2.0,
        detach_reset=False,
        surrogate_function=None,
        step_mode="s",
        **kwargs
    ):
        super().__init__()
        self.v_threshold = float(v_threshold)
        self.v_reset = float(v_reset)
        self.tau = float(tau)
        self.detach_reset = bool(detach_reset)
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.register_buffer("v", torch.tensor(0.0))

    def reset_state(self):
        self.v = torch.tensor(0.0, device=self.v.device, dtype=self.v.dtype)

    def single_step_forward(self, x):
        if self.v.dim() == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        if self.tau <= 1.0:
            before = self.v + x
        else:
            before = self.v + (x - self.v) / self.tau
        spike = (before >= self.v_threshold).to(x.dtype)
        reset_spike = spike.detach() if self.detach_reset else spike
        if self.v_reset < 0.0:
            self.v = before - reset_spike * self.v_threshold
        else:
            self.v = torch.where(
                reset_spike > 0,
                torch.full_like(before, self.v_reset),
                before,
            )
        return spike

    def forward(self, x):
        if self.step_mode == "m":
            outputs = []
            for index in range(x.shape[0]):
                outputs.append(self.single_step_forward(x[index]))
            return torch.stack(outputs)
        return self.single_step_forward(x)


def reset_custom_stateful_lif_modules(module):
    for child in module.modules():
        if isinstance(child, CustomStatefulIFNode):
            child.reset_state()


def load_workload_namespace(project_root: Path) -> Dict[str, Any]:
    source_path = project_root / "benchmarks" / "validate_kairos_baselines.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("class SingleStepModeLoopWrapper")
    end = source.index("\ndef build_placeholder_values", start)
    model_source = source[start:end]
    namespace = {
        "__name__": "chronos_nimble_workloads",
        "torch": torch,
        "nn": nn,
        "F": F,
        "functional": functional,
        "layer": layer,
        "neuron": neuron,
        "surrogate": surrogate,
        "spiking_resnet": spiking_resnet,
        "spiking_vgg": spiking_vgg,
        "CustomStatefulIFNode": CustomStatefulIFNode,
        "reset_custom_stateful_lif_modules": reset_custom_stateful_lif_modules,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
    }
    exec(compile(model_source, str(source_path), "exec"), namespace)
    return namespace
