import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from scipy.stats import wasserstein_distance
from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import *

from utils import (
    quantile_gaussian,
    build_dataset_delta,
    build_uniform_bump_dataset,
    plot_trajectories_and_hist,
    MergedMLP,
)
from fm import (
    MyAlphaTTargetConditionalFlowMatcher,
    AlphaTConditionalFlowMatcher,
    MyExactOptimalTransportConditionalFlowMatcher,
)


def train(config):
    save_path = config["save_path"]
    save_path = os.path.join("./results/", save_path)
    # first check if the file exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
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
    # y = build_dataset_delta(n_points, n_delta=n_delta).to(device)
    y = build_uniform_bump_dataset(n_points).to(device)
    #plot histogram of y
    fig = plt.figure()
    plt.hist(y.cpu().numpy(), bins=100)
    fig.savefig(os.path.join(save_path, "histogram.png"))
    plt.close(fig)
    # y_val = build_dataset_delta(n_points_val, n_delta=n_delta).to(device)
    y_val = build_uniform_bump_dataset(n_points_val).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # reducelronplateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # train
    losses = []
    for epoch in range(epochs):
        loss_epoch = 0
        for i in range(0, len(y), batch_size):
            y_batch = y[i : i + batch_size].view(-1, 1)
            x0 = torch.randn(len(y_batch), 1).to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, y_batch)
            # print(t.shape, xt.shape, ut.shape)
            vt = model(inputs=xt,flow_time=t[:, None])
            loss = criterion(vt, ut)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        loss_epoch /= len(y) / batch_size
        scheduler.step(loss_epoch)
        print(f"Epoch {epoch}: Loss {loss_epoch}")

        # validation
        with torch.no_grad():
            loss_val = 0
            for i in range(0, len(y_val), batch_size):
                y_batch = y_val[i : i + batch_size].view(-1, 1)
                x0 = quantile_points[i : i + batch_size].view(-1, 1)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, y_batch)
                vt = model(inputs=xt, flow_time=t[:, None])
                loss = criterion(vt, ut)
                loss_val += loss.item()
           
            val_loss = loss_val / (len(y_val) / batch_size)
            print(f"Epoch {epoch}: Val Loss {val_loss}")
        
            # sample and compute the wasserstein distance
            t = torch.linspace(0, 1, timesteps).to(device)
            samples_list = []
            for i in range(0, len(quantile_points), 10000):
                samples = odeint(lambda t, x: model(inputs=x, flow_time= t.expand(x.shape[0], 1),), quantile_points.view(-1,1), t, method="euler").cpu().numpy()
                samples_list.append(samples[-1,:])
                
            samples = np.concatenate(samples_list, axis=0)
                
            ws = wasserstein_distance(
                y_val.squeeze().cpu().numpy(),
                samples.squeeze(),
            )

        losses.append((loss_epoch, val_loss, ws))
        
        # plot 1d hist of y_val vs samples[-1]
        fig = plt.figure()
        # use the same binning
        bins = np.histogram_bin_edges(y_val.cpu().numpy(), bins=100)
        plt.hist(y_val.cpu().numpy(), bins=bins, alpha=0.5, label="y_val")
        plt.hist(samples[-1,:].squeeze(), bins=bins, alpha=0.5, label="samples")
        plt.legend()
        fig.savefig(os.path.join(save_path, f"hist_epoch_{epoch}.png"))
        plt.close(fig)

        # also log the two loss values in a csv in the save_path

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
            
    # at the end of training, plot losses (just val and train) and wasserstein distance in another figure
    losses = np.array(losses)
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    #log and not log
    ax[0].plot(losses[:, 0], label="Train loss")
    ax[0].plot(losses[:, 1], label="Val loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_yscale("log")
    ax[1].plot(losses[:, 0], label="Train loss")
    ax[1].plot(losses[:, 1], label="Val loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    plt.savefig(os.path.join(save_path, "losses.png"))
    plt.close(fig)
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    ax[0].plot(losses[:, 2], label="Wasserstein distance")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Wasserstein distance")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].plot(losses[:, 2], label="Wasserstein distance")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Wasserstein distance")
    ax[1].legend()
    plt.savefig(os.path.join(save_path, "wasserstein.png"))
    plt.close(fig)
    
    
    
        
if __name__ == "__main__":
    # read all the .yaml configs in the config folder
    for config in os.listdir("./configs"):
        with open(os.path.join("./configs", config), "r") as f:
            config = yaml.safe_load(f)
        train(config)