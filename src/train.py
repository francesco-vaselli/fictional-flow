import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance
from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import *

from utils import (
    quantile_gaussian,
    build_dataset_delta,
    plot_trajectories_and_hist,
    MergedMLP,
)
from fm import (
    MyAlphaTTargetConditionalFlowMatcher,
    MyExactOptimalTransportConditionalFlowMatcher,
)


def train(config):
    save_path = config["save_path"]
    save_path = os.path.join("./results/", save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    matching_type = config["cfm"]["matching_type"]
    sigma = config["cfm"]["sigma"]

    if matching_type == "Target":
        FM = MyTargetConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "AlphaT":
        FM = AlphaTConditionalFlowMatcher(sigma=sigma, alpha=config["cfm"]["alpha"])
    elif matching_type == "AlphaTTarget":
        FM = MyAlphaTTargetConditionalFlowMatcher(
            sigma=sigma, alpha=config["cfm"]["alpha"]
        )
    elif matching_type == "Default":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "MyExactOptimal":
        FM = MyExactOptimalTransportConditionalFlowMatcher(
            sigma=sigma, sampler_method=config["cfm"]["sampler_method"]
        )
    elif matching_type == "SchrodingerBridge":
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError("Matching type not found")

    model = MergedMLP(
        in_shape=1,
        out_shape=1,
        context_features=None,
        hidden_sizes=config["hidden_sizes"],
        time_varying=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    n_points = config["n_points"]
    n_points_val = config["n_points_val"]
    gaussian_points = config["gaussian_points"]
    n_delta = config["n_delta"]
    timesteps = config["timesteps"]

    # Generate quantile points from a Gaussian distribution
    quantile_points = torch.tensor(quantile_gaussian(n_points_val)).to(device).float()
    small_quantile_points = torch.tensor(quantile_gaussian(gaussian_points)).to(device).float()

    # Generate the dataset
    y = build_dataset_delta(n_points, n_delta=n_delta).to(device)
    #plot histogram of y
    fig = plt.figure()
    plt.hist(y.cpu().numpy(), bins=100)
    fig.savefig(os.path.join(save_path, "histogram.png"))
    plt.close(fig)
    y_val = build_dataset_delta(n_points_val, n_delta=n_delta).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    losses = []
    for epoch in range(epochs):
        loss_epoch = 0
        for i in range(0, len(y), batch_size):
            y_batch = y[i : i + batch_size]
            x0 = torch.randn(len(y_batch), 1).to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, y_batch)
            #print(t.shape, xt.shape, ut.shape)
            vt = model(inputs=xt,flow_time=t[:, None])
            loss = criterion(vt, ut)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        loss_epoch /= len(y) / batch_size
        print(f"Epoch {epoch}: Loss {loss_epoch}")

        # validation
        with torch.no_grad():
            t, xt, ut = FM.sample_location_and_conditional_flow(quantile_points.view(-1,1), y_val)
            vt = model(inputs=xt, flow_time=t[:, None])
            val_loss = criterion(vt, ut)
            print(f"Epoch {epoch}: Val Loss {val_loss}")
        
            # sample and compute the wasserstein distance
            t = torch.linspace(0, 1, timesteps).to(device)
            samples = odeint(lambda t, x: model(inputs=x, flow_time= t.expand(x.shape[0], 1),), quantile_points.view(-1,1), t, method="euler").cpu().numpy()
            
            ws = wasserstein_distance(
                y_val.squeeze().cpu().numpy(),
                samples[-1,:].squeeze(),
            )

        losses.append((loss_epoch, val_loss, ws))

        # also log the two loss values in a csv in the save_path
        # first check if the file exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # open the file in the correct way
        if epoch == 0:
            with open(os.path.join(save_path, "losses.csv"), "w") as f:
                f.write("train_loss,val_loss,ws\n")
                f.write(f"{loss_epoch},{val_loss},{ws}\n")
        else:
            with open(os.path.join(save_path, "losses.csv"), "a") as f:
                f.write(f"{loss_epoch},{val_loss},{ws}\n")
                
        # sample small quantile points and plot the trajectories
        with torch.no_grad():
            samples_small = odeint(lambda t, x: model(inputs=x, flow_time= t.expand(x.shape[0], 1)), small_quantile_points.view(-1,1), t, method="euler").cpu().numpy()
            fig, _ = plot_trajectories_and_hist(samples, samples_small, n=100, every_n=1)
            
            fig.savefig(os.path.join(save_path, f"epoch_{epoch}.png"))
            plt.close(fig)
        
if __name__ == "__main__":
    config = {
        "save_path": "results",
        "cfm": {
            "matching_type": "AlphaTTarget",
            "sigma": 0.0001,
            "alpha": 1,
            "sampler_method": "default",
        },
        "hidden_sizes": [32, 64, 128, 64, 32],
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 128,
        "n_points": 100000,
        "n_points_val": 10000,
        "gaussian_points": 100,
        "n_delta": 4,
        "timesteps": 100,
    }
    train(config)