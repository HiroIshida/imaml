import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.nn import Module


def hessian_free_cg(
    fun_Ax: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, x: torch.Tensor, n_iter: int
) -> torch.Tensor:
    # Algorithm 5.2 of Nocedal and Wright
    # NOTE: reason why Ax is a function is that computing the hessian (A)
    # is quite expensive, but usually computing the hessian-vector product
    # is much cheaper. So we treat this as a black-box module.
    r = fun_Ax(x) - b
    p = -r
    for k in range(n_iter):
        alpha = (r @ r) / (p @ fun_Ax(p))
        x = x + alpha * p
        r_new = r + alpha * fun_Ax(p)
        beta = (r_new @ r_new) / (r @ r)
        p = -r_new + beta * p
        r = r_new
    return x


LossFunction = Callable[[Module, Any], torch.Tensor]


def compute_regularized_loss(
    model: Module, model_original: Module, data: Any, fn_loss: LossFunction, lambda_param: float
) -> torch.Tensor:

    loss = fn_loss(model, data)
    for p, p_original in zip(model.parameters(), model_original.parameters()):
        loss += 0.5 * lambda_param * (p - p_original).pow(2).sum()
    return loss


def compute_regularized_grad(
    model: Module, model_original: Module, data: Any, fn_loss: LossFunction, lambda_param: float
) -> torch.Tensor:

    for p in model.parameters():
        p.grad = None
    loss = compute_regularized_loss(model, model_original, data, fn_loss, lambda_param)
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    grad = torch.cat([g.view(-1) for g in grads])

    for p in model.parameters():
        p.grad = None
    return grad


def get_slided_model(model: nn.Module, delta: torch.Tensor) -> nn.Module:
    model_new = copy.deepcopy(model)
    head = 0
    for p in model_new.parameters():
        p.data = p.data - delta[head : head + p.numel()].view(p.shape)
        head += p.numel()
    return model_new


def compute_meta_gradient(
    model_optimized: nn.Module,
    model_original,
    data,
    loss_fn: LossFunction,
    lambda_param: float,
    n_iter: int = 5,
) -> torch.Tensor:
    # Line 4-5 of Alg. 2 of the paper.
    # instead of delta', we use n_iter of hessian_free_cg to stop the iteration.

    def fun_Ax(vector: torch.Tensor) -> torch.Tensor:
        # (I + 1/lambda * Hess) * vector
        grad = compute_regularized_grad(
            model_optimized, model_original, data, loss_fn, lambda_param
        )
        eps = 1e-4
        model_eps = get_slided_model(model_optimized, eps * vector)
        grad_eps = compute_regularized_grad(model_eps, model_original, data, loss_fn, lambda_param)
        vector_norm = torch.norm(vector)
        return vector + (grad_eps - grad) / (eps * vector_norm) / lambda_param

    b = compute_regularized_grad(model_optimized, model_original, data, loss_fn, lambda_param)
    g_init_guess = torch.randn_like(b)
    g = hessian_free_cg(fun_Ax, b, g_init_guess, n_iter=n_iter)  # Ax = b
    return g


@dataclass
class ImamlConfig:
    n_outer: int
    lr_outer: float
    lambda_param: float
    minibatch_size: int


def imaml_loop(
    model: nn.Module,
    dataset_iterator: Iterator,
    loss_fn: LossFunction,
    algorithm: Callable[[Module, Any, LossFunction], Module],
    imaml_config: ImamlConfig,
) -> None:
    model_current = copy.deepcopy(model)
    for _ in range(imaml_config.n_outer):
        grad_list = []
        for _ in range(imaml_config.minibatch_size):
            data = next(dataset_iterator)
            loss_here = partial(
                compute_regularized_loss,
                model_original=model_current,
                fn_loss=loss_fn,
                lambda_param=imaml_config.lambda_param,
            )
            algorithm(model_current, data, loss_here)
            grad = compute_meta_gradient(
                modeoptimized, model_current, data, loss_fn, lambda_param=imaml_config.lambda_param
            )
            grad_list.append(grad)
        grad_mean = torch.stack(grad_list).mean(dim=0)
        model_current = get_slided_model(model_current, imaml_config.lr_outer * grad_mean)
