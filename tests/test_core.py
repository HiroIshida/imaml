import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam

from imaml.core import (
    compute_meta_gradient,
    compute_regularized_loss,
    get_slided_model,
    hessian_free_cg,
)


def raw_loss_function(model, dataset):
    x, y = dataset
    y_hat = model(x)
    criterion = nn.MSELoss()
    return criterion(y_hat, y)


def datasets(n_pts=100):

    for _ in range(100):
        A = np.random.uniform(1, 2)
        omega = np.random.uniform(2, 4)
        phi = np.random.uniform(0, 2 * np.pi)
        x = torch.linspace(-np.pi, np.pi, n_pts).view(-1, 1)
        y = torch.sin(omega * x - phi) * A
        yield x, y


def linear_datasets(n_pts=10):
    for _ in range(100):
        A = np.random.uniform(1, 2)
        b = np.random.uniform(2, 4)
        x = torch.linspace(-np.pi, np.pi, n_pts).view(-1, 1)
        y = A * x + b
        yield x, y


def adam_algorithm(model, samples: Tuple[torch.Tensor, ...], loss_function):
    model = copy.deepcopy(model)
    optimizer = Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        loss = loss_function(model, samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 300 == 0:
            print(f"Epoch [{epoch+1}/3000], Loss: {loss.item()}")
    return model


def lbfgs_algorithm(model, samples: Tuple[torch.Tensor, ...], loss_function):
    model = copy.deepcopy(model)
    optimizer = LBFGS(
        model.parameters(),
        lr=0.01,
        max_iter=20,
        max_eval=25,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, samples)
        loss.backward()
        return loss

    for epoch in range(50):
        optimizer.step(closure)
        loss = loss_function(model, samples)
    print(loss)
    return model


def test_hessian_free_cg():
    n = 10
    A = np.random.randn(n, n)
    x = np.random.randn(n)
    b = A @ x
    fun_Ax = lambda x: A @ x
    hessian_free_cg(fun_Ax, b, np.zeros(n), 10)


if __name__ == "__main__":
    # model_original = nn.Sequential(
    #     nn.Linear(1, 5),
    #     nn.ReLU(),
    #     nn.Linear(5, 1),
    # )
    model_original = nn.Sequential(
        nn.Linear(1, 1),
    )
    param_original = torch.cat([p.view(-1) for p in model_original.parameters()])
    print(param_original)
    dataset = next(linear_datasets())

    lambda_param = 0.1

    def loss_wrap(model, dataset):
        return compute_regularized_loss(
            model, model_original, dataset, raw_loss_function, lambda_param
        )

    model_opt = lbfgs_algorithm(model_original, dataset, loss_wrap)
    grad = compute_meta_gradient(
        model_opt, model_original, dataset, raw_loss_function, lambda_param, n_iter=20
    )
    print(grad[0])

    eps = 1e-3
    slide = param_original * 0.0
    slide[0] += eps
    model_slided = get_slided_model(model_original, slide)
    model_opt_eps = lbfgs_algorithm(model_slided, dataset, loss_wrap)
    param_eps = torch.cat([p.view(-1) for p in model_opt_eps.parameters()])
    grad_numel = (loss_wrap(model_opt_eps, dataset) - loss_wrap(model_opt, dataset)) / eps
    print(grad_numel)

    # import matplotlib.pyplot as plt
    # plt.plot(x.detach().numpy(), y.detach().numpy(), 'o')
    # plt.plot(x.detach().numpy(), model_opt(x).detach().numpy(), 'o')
    # plt.show()
