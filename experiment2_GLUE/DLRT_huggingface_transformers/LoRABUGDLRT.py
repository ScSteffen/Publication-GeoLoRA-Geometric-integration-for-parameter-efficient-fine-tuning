import torch
import torch.nn as nn
import math


class LoRABUGDLRT(nn.Module):
    def __init__(
        self, original_linear, rank, tau, alpha=16, max_rank=32, lora_dropout=0.0
    ):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where
           U, S, V represent the facorized weight W
        Args:
            rank: initial rank of factorized weight
        """
        # construct parent class nn.Module
        super(LoRABUGDLRT, self).__init__()

        self.original_linear = original_linear
        self.original_linear.requires_grad = False
        # set rank and truncation tolerance for parallel LoRA

        self.r = rank
        self.tol = tau
        self.rmax = max_rank
        self.rmin = 2
        # Scaling factor
        self.alpha = alpha
        self.scaling = self.alpha / self.r  # probably not needed for dlrt

        self.lora_U = nn.Parameter(
            torch.linalg.qr(
                torch.randn(original_linear.in_features, self.rmax), "reduced"
            )[0],
            requires_grad=True,
        )
        self.lora_V = nn.Parameter(
            torch.linalg.qr(
                torch.randn(original_linear.out_features, self.rmax), "reduced"
            )[0],
            requires_grad=True,
        )

        self.lora_S = nn.Parameter(
            torch.zeros(self.rmax, self.rmax),
            requires_grad=True,
        )
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()

        # self.Sinv = nn.Parameter(
        #    torch.eye(self.rmax), requires_grad=False
        # )  # made parameter for multi gpu,  identity initialization for lora specifically

    def forward(self, x):
        """Returns the output of the layer. The formula implemented is output =  xW + x*U*S*V' + bias.
        Args:
            x: input to layer
        Returns:
            output of layer
        """
        # out = self.original_linear(x) + self.scaling * (
        #   x @ self.lora_U[:, : self.r] @ self.lora_V[:, : self.r].T
        # )
        out = self.original_linear(x) + self.scaling * (
            (
                (self.lora_dropout_layer(x) @ self.lora_U[:, : self.r])
                @ self.lora_S[: self.r, : self.r]
            )
            @ self.lora_V[:, : self.r].T
        )
        return out

    @torch.no_grad()
    def step(
        self, learning_rate, dlrt_step="basis", momentum=0.0, weight_decay=0.0
    ) -> None:

        U_view = self.lora_U  # just a view, no copy
        V_view = self.lora_V  # just a view, no copy
        S_view = self.lora_S

        if dlrt_step == "basis":
            r1 = min(self.rmax, 2 * self.r)

            U1, _ = torch.linalg.qr(
                torch.cat((U_view[:, : self.r], U_view.grad[:, : self.r]), 1),
                "reduced",
            )

            V1, _ = torch.linalg.qr(
                torch.cat((V_view[:, : self.r], V_view.grad[:, : self.r]), 1),
                "reduced",
            )

            # M = U1[:, :r1].T @ U_view[:, : self.r]
            # N = V_view[:, : self.r].T @ V1[:, :r1]
            # Project coefficients
            S_view.data[:r1, :r1] = (
                (U1[:, :r1].T @ U_view[:, : self.r])
                @ S_view[: self.r, : self.r]
                @ (V_view[:, : self.r].T @ V1[:, :r1])
            )

            # update basis
            U_view.data[:, :r1] = U1[:, :r1]
            V_view.data[:, :r1] = V1[:, :r1]

            self.r = r1

            S_view.grad.zero_()
            U_view.grad.zero_()
            V_view.grad.zero_()

        elif dlrt_step == "coefficients":
            # self.bS[: self.r, : self.r] = (
            #    momentum * self.bS[: self.r, : self.r]
            #    + S_view.grad[: self.r, : self.r]
            #    + weight_decay * S_view.data[: self.r, : self.r]
            # )
            S_view.data[: self.r, : self.r] = (
                S_view[: self.r, : self.r]
                - learning_rate * S_view.grad[: self.r, : self.r]
            )
            S_view.grad.zero_()
            U_view.grad.zero_()
            V_view.grad.zero_()

        elif dlrt_step == "truncate":
            # truncate to new rank
            self.Truncate()
        else:
            print("Wrong step defined: ", dlrt_step)

    @torch.no_grad()
    def Truncate(self):
        """Truncates the weight matrix to a new rank"""
        P, d, Q = torch.linalg.svd(self.lora_S[: self.r, : self.r])

        tol = self.tol * torch.linalg.norm(d)
        r1 = self.r
        for j in range(0, self.r):
            tmp = torch.linalg.norm(d[j : self.r])
            if tmp < tol:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, 2)

        # update s
        self.lora_S.data[:r1, :r1] = torch.diag(d[:r1])
        # self.Sinv[:r1, :r1] = torch.diag(1.0 / d[:r1])

        # update u and v
        self.lora_U.data[:, :r1] = self.lora_U[:, : self.r] @ P[:, :r1]
        self.lora_V.data[:, :r1] = self.lora_V[:, : self.r] @ Q.T[:, :r1]
        self.r = int(r1)
        self.scaling = self.alpha / self.r

    def basis_grad_zero(self):
        self.lora_U.grad.zero_()
        self.lora_V.grad.zero_()
