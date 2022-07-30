import enum
import os
import logging
import io
from random import random
import warnings
from matplotlib.colors import ListedColormap

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import wandb


logger = logging.getLogger("UTILS")
warnings.filterwarnings(action='ignore', category=FutureWarning)
sns.set()

# def is_debug():
#     return str2bool(os.environ.get("DEBUG", "false"))


def str2bool(s):
    if s.lower() in ['true', 'yes', 'y']:
        return True
    elif s.lower() in ['false', 'no', 'n']:
        return False
    else:
        raise RuntimeError('Boolean value expected')


DEBUG = str2bool(os.environ.get("DEBUG", "false"))
# Whether to immediately commit results to wandb, running
# many runs in parallel may result in rate limit error.
# Setting WANDB_COMMIT to False will only commit result after
# run has finished
# WANDB_COMMIT = str2bool(os.environ.get("WANDB_COMMIT", "true"))
WANDB_COMMIT = False
SHOW_PLOT = str2bool(os.environ.get("SHOW_PLOT", "false"))


def mpl_fig_to_wandb_image(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")

    im = Image.open(img_buf)

    return wandb.Image(im)


class ReconstructImages:

    def __init__(self, num_images = 5, mean_only = True):
        self.num_images = num_images
        self.mean_only = mean_only

    def __call__(
        self, stage, model, data_loader, step,
    ) -> None:
        model.eval()

        dataset = data_loader.dataset
        indices = torch.randperm(len(dataset))[: self.num_images]

        with torch.no_grad():
            x = torch.stack([dataset[idx][0] for idx in indices], dim=0)

            x = x.cuda()

            x_hat = model.reconstruct(x, mean_only=self.mean_only)
            x_hat = x_hat.view_as(x)

        xs = torch.cat([x, x_hat])
        grid = make_grid(
            xs, nrow=self.num_images, padding=1, pad_value=0.5
        )
        image = wandb.Image(grid, caption="Top: Ground-truth, Bottom: Reconstruction")

        if not DEBUG:
            wandb.log({f"Reconstruction/{stage}": image}, step=step, commit=WANDB_COMMIT)


class PlotLatentSpace:

    def __init__(self, num_batches = None, quantiles = [0.025, 0.975], size_recon = 12):
        self.num_batches = num_batches
        self.quantiles = quantiles
        self.size_recon = size_recon

    def __call__(self, stage, model, data_loader, step):
        model.eval()

        zs, ys = [], []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.cuda()

                z = model.encode(x, mean_only=True)

                zs.append(z.cpu().numpy())
                ys.append(y.numpy())

                if self.num_batches is not None and batch_idx == self.num_batches - 1:
                    break

        zs = np.concatenate(zs, axis=0)
        ys = np.concatenate(ys, axis=0)

        # Create histogram of each latent variable
        zs_mean = np.mean(zs, axis=0)
        zs_std = np.std(zs, axis=0)
        assert len(zs_mean) == zs.shape[-1] and len(zs_std) == zs.shape[-1]

        ncols = min(3, zs.shape[-1])
        nrows = zs.shape[-1] // ncols
        nrows += 0 if zs.shape[-1] % ncols == 0 else 1
        fig = plt.Figure(figsize=(9, nrows * 3))
        for latent_dim in range(zs.shape[-1]):
            ax = fig.add_subplot(nrows, ncols, latent_dim + 1)

            ax.hist(zs[:, latent_dim])
            mean = zs_mean[latent_dim]
            std = zs_std[latent_dim]
            ax.set_title(f"{mean:.3f} +/- {std:.3f}")
            ax.autoscale()

        fig.tight_layout()

        logs = {f"{stage} z space/Histogram": mpl_fig_to_wandb_image(fig)}

        plt.close(fig)

        use_compression = zs.shape[-1] > 2

        # Create scatter plot using t-SNE with ground-truth labels
        tsne_error = False
        if use_compression:
            try:
                tsne = TSNE(n_components=2, init="pca", random_state=0)
                zs_tsne = tsne.fit_transform(zs)
            except Exception:
                tsne_error = True
        else:
            zs_tsne = zs
        
        if not tsne_error:
            color_palette = sns.color_palette(n_colors=len(np.unique(ys)))
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            scatter = ax.scatter(x=zs_tsne[:, 0], y=zs_tsne[:, 1], c=ys, cmap=ListedColormap(color_palette))
            ax.legend(*scatter.legend_elements())
            ax.autoscale()
            fig.tight_layout()
            
            logs[f"{stage} z space/t-SNE"] = mpl_fig_to_wandb_image(fig)
            plt.close(fig)

        # If latent space has more than 2 components
        # use the 2 main principal components
        pca_error = False
        if use_compression:
            pca = PCA(n_components=2)
            try:
                zs_pca = pca.fit_transform(zs)
            except Exception as e:
                logger.error(e)
                pca_error = True
        else:
            zs_pca = zs

        if not pca_error:
            # Traverse latent space
            z_quantiles = np.quantile(zs_pca, self.quantiles, axis=0).T
            zs_traverse = []
            for y in np.linspace(*z_quantiles[1], self.size_recon):
                for x in np.linspace(*z_quantiles[0], self.size_recon):
                    zs_traverse.append([x, y])

            # Transforms back to correct latent size
            if use_compression:
                zs_traverse = pca.inverse_transform(zs_traverse)

            with torch.no_grad():
                zs_traverse = torch.tensor(zs_traverse).float()
                zs_traverse = zs_traverse.cuda()

                x_hat_traverse = model.decode(zs_traverse)

                grid = make_grid(x_hat_traverse, nrow=self.size_recon, padding=0)

            color_palette = sns.color_palette(n_colors=len(np.unique(ys)))
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            scatter = ax.scatter(x=zs_pca[:, 0], y=zs_pca[:, 1], c=ys, cmap=ListedColormap(color_palette))
            ax.legend(*scatter.legend_elements())
            ax.autoscale()
            fig.tight_layout()
            
            logs[f"{stage} z space/PCA"] = mpl_fig_to_wandb_image(fig)
            plt.close(fig)

            logs[f"{stage} z space/traversal"] = wandb.Image(grid)

        # Wandb Logging
        if not DEBUG:
            wandb.log(
                logs, 
                step=step, 
                commit=WANDB_COMMIT
            )
        
        if SHOW_PLOT:
            plt.show()

