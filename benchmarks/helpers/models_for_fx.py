import torch
import torch.nn as nn
from spikingjelly.activation_based import base, functional, surrogate
from spikingjelly.activation_based.model import spiking_resnet

import runtime.snn_custom_ops  # noqa: F401 - registers torch.library ops


class CustomStatefulIFNode(nn.Module, base.StepModule):
    def __init__(
        self,
        v_threshold=1.0,
        v_reset=0.0,
        tau=2.0,
        detach_reset=False,
        surrogate_function=None,
        step_mode="s",
        **kwargs,
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau = tau
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        self.step_mode = step_mode
        self.register_buffer("v", torch.tensor(0.0), persistent=False)
        self.register_buffer("_reset_v", torch.tensor(0.0), persistent=False)

    def reset_state_if_needed(self, x):
        if self.v.dim() == 0 or self.v.shape != x.shape or self.v.device != x.device or self.v.dtype != x.dtype:
            self.v = torch.zeros_like(x)

    def reset_state(self):
        if self._reset_v.device != self.v.device or self._reset_v.dtype != self.v.dtype:
            self._reset_v = torch.tensor(0.0, device=self.v.device, dtype=self.v.dtype)
        self.v = self._reset_v

    def single_step_forward(self, x):
        self.reset_state_if_needed(x)
        spike, v_next = torch.ops.snn_custom.lif_forward_state.default(
            x,
            self.v,
            float(self.v_threshold),
            float(self.v_reset),
            float(self.tau),
            bool(self.detach_reset),
        )
        self.v = v_next
        return spike

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward(
        x_seq: torch.Tensor,
        v: torch.Tensor,
        v_threshold: float,
        v_reset: float,
        tau: float,
        detach_reset: bool,
    ):
        spikes = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            if tau <= 1.0:
                v_before_spike = v + x_seq[t]
            else:
                v_before_spike = v + (x_seq[t] - v) / tau
            spike = (v_before_spike >= v_threshold).to(x_seq)
            spike_for_reset = spike.detach() if detach_reset else spike
            if v_reset < 0.0:
                v = v_before_spike - spike_for_reset * v_threshold
            else:
                v = torch.where(
                    spike_for_reset > 0.0,
                    torch.full_like(v_before_spike, v_reset),
                    v_before_spike,
                )
            spikes[t] = spike
        return spikes, v

    def multi_step_forward(self, x_seq):
        if not self.training:
            self.reset_state_if_needed(x_seq[0])
            spikes, self.v = self.jit_eval_multi_step_forward(
                x_seq,
                self.v,
                float(self.v_threshold),
                float(self.v_reset),
                float(self.tau),
                bool(self.detach_reset),
            )
            return spikes
        spikes = torch.empty_like(x_seq)
        for t in range(x_seq.shape[0]):
            spikes[t] = self.single_step_forward(x_seq[t])
        return spikes

    def forward(self, x):
        if self.step_mode == "m":
            if x.dim() < 3:
                raise ValueError(
                    f"CustomStatefulIFNode in step_mode='m' expects [T, N, ...], got shape={tuple(x.shape)}"
                )
            return self.multi_step_forward(x)
        return self.single_step_forward(x)

class TinyConvStatefulLIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.lif = CustomStatefulIFNode(v_threshold=1.0, v_reset=0.0, tau=2.0)

    def forward(self, x):
        return self.lif(self.conv(x))


class SNNCustomStatefulResNet18(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T
        self.layer = spiking_resnet.spiking_resnet18(
            pretrained=False,
            spiking_neuron=CustomStatefulIFNode,
            surrogate_function=surrogate.ATan(),
        )
        functional.set_step_mode(self, step_mode="s")

    def forward(self, x):
        out_spikes_counter = 0
        for _ in range(self.T):
            out_spikes_counter = out_spikes_counter + self.layer(x)
        return out_spikes_counter / self.T


def build_model(name: str, T: int):
    if name == "tiny-stateful":
        return TinyConvStatefulLIF()
    if name == "resnet18":
        return SNNCustomStatefulResNet18(T=T)
    raise ValueError(f"unsupported model: {name}")


def reset_custom_stateful_lif_modules(model: nn.Module):
    for module in model.modules():
        if hasattr(module, "reset_state"):
            module.reset_state()
