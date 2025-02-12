import torch
import torch.nn as nn
import math


class LoRAParallelDLRT(nn.Module):
    def __init__(
        self, original_linear, rank, tau, alpha=16, max_rank=32, lora_dropout=0.0
    ):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where
           U, S, V represent the facorized weight W
        Args:
            rank: initial rank of factorized weight
        """
        # construct parent class nn.Module
        super(LoRAParallelDLRT, self).__init__()

        self.original_linear = original_linear

        # set rank and truncation tolerance for parallel LoRA

        self.r = rank
        self.tol = tau
        self.rmax = min(
            original_linear.out_features, min(original_linear.in_features, max_rank)
        )
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.rmin = 2
        self.dims = [self.out_features,self.in_features]
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

        self.Sinv = nn.Parameter(
            torch.eye(self.rmax), requires_grad=False
        )  # made parameter for multi gpu,  identity initialization for lora specifically

        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()

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
    def step(self, learning_rate, weight_decay):
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
        """
        r = self.r

        r1 = min(2 * r, self.rmax)

        U_view = self.lora_U  # just a view, no copy
        V_view = self.lora_V  # just a view, no copy
        S_view = self.lora_S
        Sinv_view = self.Sinv

        # perform K-step
        # gradient modification
        U_view.grad[:, :r] = U_view.grad[:, :r] @ Sinv_view[:r, :r]
        U1, _ = torch.linalg.qr(
            torch.cat((U_view[:, :r], U_view.grad[:, :r]), 1), "reduced"
        )

        # if torch.isnan(U1).any():
        #    print("U1 tensor contains nans")
        #    exit(1)

        # perform L-step
        # gradient modification
        V_view.grad[:, :r] = V_view.grad[:, :r] @ Sinv_view[:r, :r]
        V1, _ = torch.linalg.qr(
            torch.cat((V_view[:, :r], V_view.grad[:, :r]), 1), "reduced"
        )

        # if torch.isnan(V1).any():
        #    print("V1 tensor contains nans")
        #    exit(1)

        # set up augmented S matrix
        try:
            S_view.data[r:r1, :r] = U1[:, r:r1].T @ (
                U_view[:, :r] @ S_view[:r, :r] - learning_rate * U_view.grad[:, :r]
            )
        except:
            print("error SK", r, r1, U_view.shape, U1.shape)
            print("end")
            exit(1)
        try:
            S_view.data[:r, r:r1] = (
                V_view[:, :r] @ S_view[:r, :r].T - learning_rate * V_view.grad[:, :r]
            ).T @ V1[:, r:r1]
        except:
            print("error SL", r, r1, V_view.shape, V1.shape)
            print("end")
            exit(1)

        # needs to go after SK and SL, since S gets updated here and SK and SL needs old S
        S_view.data[:r, :r] = S_view[:r, :r] - learning_rate * S_view.grad[:r, :r]
        S_view.data[r:r1, r:r1] *= 0  # = torch.zeros((r, r))

        U_view.data[:, r:r1] = U1[:, r:r1]  # torch.cat((U0, U1[:, r:r1]), 1)
        V_view.data[:, r:r1] = V1[:, r:r1]  # torch.cat((V0, V1[:, r:r1]), 1)
        # print(self.r)
        self.r = r1

        # if torch.isnan(S_view.data[: self.r, : self.r]).any():
        #    print("S tensor contains nans")
        #    exit(1)
        # if torch.isnan(U_view.data[:, : self.r]).any():
        #    print("U tensor contains nans")
        # if torch.isnan(V_view.data[:, : self.r]).any():
        #    print("V tensor contains nans")

        S_view.grad.zero_()
        U_view.grad.zero_()
        V_view.grad.zero_()

        # self.Truncate()

    @torch.no_grad()
    def Truncate(self):
        # print(self.r)
        """Truncates the weight matrix to a new rank"""
        P, d, Q = torch.linalg.svd(self.lora_S[: self.r, : self.r])

        tol = self.tol * torch.linalg.norm(d)
        # print(tol)
        r1 = self.r
        for j in range(0, self.r):
            tmp = torch.linalg.norm(d[j : self.r])
            # print(tmp)
            if tmp < tol:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, self.rmin)

        # update s
        self.lora_S.data[:r1, :r1] = torch.diag(d[:r1])
        self.Sinv[:r1, :r1] = torch.diag(1.0 / d[:r1])

        # update u and v
        self.lora_U.data[:, :r1] = self.lora_U[:, : self.r] @ P[:, :r1]
        self.lora_V.data[:, :r1] = self.lora_V[:, : self.r] @ Q.T[:, :r1]
        self.r = int(r1)
        self.scaling = self.alpha / self.r

        # self.lora_U.contiguous()
        # self.lora_V.contiguous()
        # self.lora_S.contiguous()

    def basis_grad_zero(self):
        self.lora_U.grad.zero_()
        self.lora_V.grad.zero_()

    def coeff_grad_zero(self):
        self.lora_S.grad.zero_()
