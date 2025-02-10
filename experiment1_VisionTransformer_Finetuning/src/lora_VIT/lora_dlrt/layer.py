# Copyright 2023-present the HuggingFace Inc. team.
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

import warnings
from typing import Any, List, Optional

import torch
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose


class AdaLoraLayer(LoraLayer):
    """
    modification of the adalora layer for which lora_E is not a diagonal matrix but a full one
    """

    # List all names of layers that may contain adapter weights
    # Note: ranknum doesn't need to be included as it is not an nn.Module
    adapter_layer_names = (
        "lora_A",
        "lora_B",
        "lora_E",
        "lora_embedding_A",
        "lora_embedding_B",
    )
    # other_param_names is defined in LoraLayer

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})

        self.P = None
        self.d = None
        self.Q = None

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(
                f"`r` should be a positive integer or 0, but the value passed is {r}"
            )

        self.r[adapter_name] = min([r, self.in_features, self.out_features])
        self.adapter_name = adapter_name
        self.rmax = min([self.in_features, self.out_features])

        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        # Right singular vectors
        r = self.rmax
        self.lora_A[adapter_name] = nn.Parameter(
            torch.randn(self.in_features, r, dtype=torch.float32)
        )
        # Singular values
        self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, r, dtype=torch.float32))
        # Left singular vectors
        self.lora_B[adapter_name] = nn.Parameter(
            torch.randn(self.out_features, r, dtype=torch.float32)
        )
        self.adapter_name = adapter_name
        ##################
        # The current rank
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        if hasattr(self.get_base_layer(), "qweight"):
            # QuantLinear
            self.to(self.get_base_layer().qweight.device)
        else:
            self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)
            self.lora_B[adapter_name].copy_(
                torch.linalg.qr(self.lora_B[adapter_name], "reduced")[0]
            )
            self.lora_A[adapter_name].copy_(
                torch.linalg.qr(self.lora_A[adapter_name], "reduced")[0]
            )
            nn.init.uniform_(self.lora_E[adapter_name], a=1, b=100)
            self.lora_E[adapter_name].copy_(
                torch.diag(torch.abs(torch.diag(0.0 * self.lora_E[adapter_name])))
            )  #### initialize at zero
            self.Einv = torch.nn.Parameter(
                torch.eye(self.lora_E[adapter_name].shape[0]), requires_grad=False
            )  #### initialize as identity

    @torch.no_grad()
    def step(self, learning_rate, tau=1e-1, adapter_name="dlrt"):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
        """
        r = self.r[adapter_name]
        r1 = min(2 * r, self.rmax)
        U_view = self.lora_A[adapter_name]  # just a view, no copy
        V_view = self.lora_B[adapter_name]
        S_view = self.lora_E[adapter_name]
        Sinv_view = self.Einv

        # perform K-step
        # gradient modification
        U_view.grad[:, :r] = U_view.grad[:, :r] @ Sinv_view[:r, :r].T
        U1, _ = torch.linalg.qr(
            torch.cat((U_view[:, :r], U_view.grad[:, :r]), 1), "reduced"
        )

        # perform L-step
        # gradient modification
        V_view.grad[:, :r] = V_view.grad[:, :r] @ Sinv_view[:r, :r]
        V1, _ = torch.linalg.qr(
            torch.cat((V_view[:, :r], V_view.grad[:, :r]), 1), "reduced"
        )

        # set up augmented S matrix

        S_view.data[r:r1, :r] = U1[:, r:r1].T @ (
            U_view[:, :r] @ S_view[:r, :r] - learning_rate * U_view.grad[:, :r]
        )
        S_view.data[:r, r:r1] = (
            V_view[:, :r] @ S_view[:r, :r].T - learning_rate * V_view.grad[:, :r]
        ).T @ V1[:, r:r1]
        # needs to go after SK and SL, since S gets updated here and SK and SL needs old S
        S_view.data[:r, :r] = S_view[:r, :r] - learning_rate * S_view.grad[:r, :r]
        S_view.data[r:r1, r:r1] *= 0

        U_view.data[:, r:r1] = U1[:, r:r1]
        V_view.data[:, r:r1] = V1[:, r:r1]

        self.r[adapter_name] = r1

        # self.Truncate(tau=tau, adapter_name=adapter_name)

    @torch.no_grad()
    def get_singular_values(self, adapter_name="dlrt"):
        r = self.r[adapter_name]
        # print(f'lora E {self.lora_E[adapter_name][:r, :r]}')
        self.P, self.d, self.Q = torch.linalg.svd(self.lora_E[adapter_name][:r, :r])
        return self.d

    @torch.no_grad()
    def budget_truncate(self, num_singular_values, adapter_name="default"):
        """Truncates the weight matrix to a new rank"""
        r = self.r[adapter_name]
        # print(f'lora E {self.lora_E[adapter_name][:r, :r]}')
        P, d, Q = self.P, self.d, self.Q

        r1 = num_singular_values
        # print(r1)

        r1 = min(r1, self.rmax)
        r1 = max(r1, 2)

        # update s
        self.lora_E[adapter_name][:r1, :r1].copy_(torch.diag(d[:r1]))
        self.Einv[:r1, :r1].copy_(torch.diag(1.0 / d[:r1]))

        # update u and v
        self.lora_B[adapter_name][:, :r1].copy_(
            torch.matmul(self.lora_B[adapter_name][:, :r], P[:, :r1])
        )
        self.lora_A[adapter_name][:, :r1].copy_(
            torch.matmul(self.lora_A[adapter_name][:, :r], Q.T[:, :r1])
        )
        self.r[adapter_name] = int(r1)

    @torch.no_grad()
    def Truncate(self, tau=1e-1, adapter_name="default"):
        """Truncates the weight matrix to a new rank"""
        r = self.r[adapter_name]
        # print(f'lora E {self.lora_E[adapter_name][:r, :r]}')
        P, d, Q = torch.linalg.svd(self.lora_E[adapter_name][:r, :r])

        # print(torch.linalg.matrix_norm(P @ torch.diag(d) @ Q.t() - self.S[:self.r, :self.r], 'fro'))
        tol = tau * torch.linalg.norm(d)
        r1 = r  # self.r
        for j in range(0, r):
            tmp = torch.linalg.norm(d[j:r])
            if tmp < tol:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, 2)

        # update s
        self.lora_E[adapter_name][:r1, :r1].copy_(torch.diag(d[:r1]))
        self.Einv[:r1, :r1].copy_(torch.diag(1.0 / d[:r1]))

        # update u and v
        self.lora_B[adapter_name][:, :r1].copy_(
            torch.matmul(self.lora_B[adapter_name][:, :r], P[:, :r1])
        )
        self.lora_A[adapter_name][:, :r1].copy_(
            torch.matmul(self.lora_A[adapter_name][:, :r], Q.T[:, :r1])
        )
        self.r[adapter_name] = int(r1)

    @torch.no_grad()
    def local_budget_truncate(self, tau=1e-1, b=5, adapter_name="default"):
        """Truncates the weight matrix to a new rank"""
        r = self.r[adapter_name]
        # print(f'lora E {self.lora_E[adapter_name][:r, :r]}')
        P, d, Q = torch.linalg.svd(self.lora_E[adapter_name][:r, :r])

        # print(torch.linalg.matrix_norm(P @ torch.diag(d) @ Q.t() - self.S[:self.r, :self.r], 'fro'))
        tol = tau * torch.linalg.norm(d)
        r1 = r  # self.r
        for j in range(0, r):
            tmp = torch.linalg.norm(d[j:r])
            if tmp < tol:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, 2)

        # update s
        self.lora_E[adapter_name][:r1, :r1].copy_(torch.diag(d[:r1]))
        self.Einv[:r1, :r1].copy_(torch.diag(1.0 / d[:r1]))

        # update u and v
        self.lora_B[adapter_name][:, :r1].copy_(
            torch.matmul(self.lora_B[adapter_name][:, :r], P[:, :r1])
        )
        self.lora_A[adapter_name][:, :r1].copy_(
            torch.matmul(self.lora_A[adapter_name][:, :r], Q.T[:, :r1])
        )
        self.r[adapter_name] = int(r1)


class SVDLinear(nn.Module, AdaLoraLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        AdaLoraLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(
                    active_adapter
                )

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return (
            transpose(
                self.lora_B[adapter] @ (self.lora_E[adapter] @ self.lora_A[adapter].T),
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
            / (self.ranknum[adapter] + 1e-5)
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_E = self.lora_E[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                ranknum = self.ranknum[active_adapter] + 1e-5
                r = self.r[active_adapter]
                x = x.to(lora_A.dtype)
                result += ((dropout(x) @ lora_A[:, :r]) @ lora_E[:r, :r]) @ lora_B[
                    :, :r
                ].T
                # * scaling
                # / ranknum

                result += self.bias

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adalora." + rep


class RankAllocator:  #### just an extra object used in Adalora, we do not use it but we keep it for consistency (it needs to be called during training to act)
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("lora_A", "%s"))
        self.name_set = sorted(self.name_set)
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set)

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int(
                (self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt
            )
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = (
                        self.beta1 * self.exp_avg_ipt[n]
                        + (1 - self.beta1) * self.ipt[n]
                    )
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n]
                        + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_budget(self, model, budget):
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_m % "lora_E"
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue(
            torch.cat(all_score),
            k=self.init_bgt - budget,
        )[0].item()

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0)
                    rank_pattern[n] = (
                        (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
                    )
        return rank_pattern

    def update_and_allocate(self, model, global_step, force_mask=False):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    key = (
                        n
                        if not is_adapter_name_truncated
                        else n.replace(f".{self.adapter_name}", "")
                    )
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    p.masked_fill_(~mask.bool(), 0.0)
