import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def quantile_gaussian(n_points):
    """
    Generate n_points quantile points from a Gaussian distribution.

    Parameters:
    n_points (int): The number of quantile points to generate.

    Returns:
    numpy.ndarray: The quantile points.
    """
    return norm.ppf(np.linspace(0.01, 0.99, n_points))


def build_dataset_delta(n_points, n_delta=4):

    if n_points // n_delta == 0:
        raise ValueError("n_points must be divisible by n_delta")

    fraction_per_delta = n_points // n_delta

    if n_delta == 4:
        y = torch.cat(
            [
                torch.full((fraction_per_delta, 1), -2),
                torch.full((fraction_per_delta, 1), -0.8),
                torch.full((fraction_per_delta, 1), 2),
                torch.full((fraction_per_delta, 1), 0.8),
            ],
            dim=0,
        ).to("cuda")

    elif n_delta == 3:
        y = torch.cat(
            [
                torch.full((fraction_per_delta, 1), -2),
                torch.full((fraction_per_delta, 1), 0),
                torch.full((fraction_per_delta, 1), 2),
            ],
            dim=0,
        ).to("cuda")

    elif n_delta == 2:
        y = torch.cat(
            [
                torch.full((fraction_per_delta, 1), -1),
                torch.full((fraction_per_delta, 1), 1),
            ],
            dim=0,
        ).to("cuda")

    else:
        raise ValueError("n_delta must be 2, 3 or 4")

    idx = torch.randperm(len(y))
    y = y[idx]

    return y


def plot_trajectories_and_hist(traj, traj_small, n=100, every_n=1):
    """
    Plot trajectories of some selected samples from noise to target.

    Parameters:
    traj (numpy.ndarray): The complete trajectory from noise to target.
    traj_small (numpy.ndarray): A subset of the trajectory.
    n (int, optional): The number of samples to plot. Default is 100.
    every_n (int, optional): Plot the intermediate points 1 time every `every_n` on their first dimension.
                             Default is 1 (plot all intermediate points).

    Returns:
    None
    """
    # do three plots: top for target, middle for flow, bottom for noise
    fig, ax = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Scatter plot of the intermediate points
    intermediate_points = traj_small[::every_n, 1:-1, 0]
    num_intermediate_points = len(intermediate_points)
    y_positions = [
        np.full(len(intermediate_points[i]), i) for i in range(num_intermediate_points)
    ]
    ax[1].scatter(
        intermediate_points,
        np.concatenate(y_positions),
        s=0.2,
        alpha=0.4,
        c="olive",
        label="Flow",
    )

    # Histogram of the target distribution
    ax[0].hist(
        traj[-1, :, 0], bins=50, density=True, alpha=0.8, color="blue", label="z(0)"
    )
    # ax[0].set_xlim(0, ax[0].get_xlim()[1])
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    # Histogram of the noise distribution
    ax[2].hist(
        traj_small[0, :n, 0],
        bins=50,
        range=[-4, 4],
        density=True,
        alpha=0.8,
        color="black",
        label="Prior sample z(S)",
    )
    ax[2].spines["top"].set_visible(False)
    ax[2].spines["right"].set_visible(False)
    ax[2].legend()

    return fig, ax


class MergedMLP(nn.Module):
    """
    a merged MLP for the context and the time varying component
    """

    def __init__(
        self,
        in_shape: int,
        out_shape: int,
        context_features: int,
        hidden_sizes: list,
        activation=F.relu,
        activate_output: bool = False,
        batch_norm: bool = False,
        dropout_probability: float = 0.0,
        time_varying: bool = False,
    ):
        super().__init__()
        self.time_varying = time_varying

        layers = []
        prev_size = in_shape + context_features if context_features else in_shape
        if time_varying:
            prev_size += 1  # Add one for the time component

        # Initialize hidden layers
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(p=dropout_probability))
            layers.append(activation)
            prev_size = size

        # Final layer
        layers.append(nn.Linear(prev_size, out_shape))
        if activate_output:
            layers.append(activation)

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor = None,
        flow_time: torch.Tensor = None,
    ):
        if context is not None:
            inputs = torch.cat((inputs, context), dim=1)
        if self.time_varying:  # and flow_time is not None:
            inputs = torch.cat((inputs, flow_time), dim=1)
        return self.network(inputs)
