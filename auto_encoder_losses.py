from torch import nn
from kmeans_pytorch import kmeans


class Losses:
    CENTROIDS = None
    RECONSTRUCTION_CRITERION = nn.MSELoss()
    N_CLUSTERS = 10
    WARMUP = 10

    def mse_k_means(X, Y, latent, iteration, alpha=0.1):
        # https://discuss.pytorch.org/t/k-means-loss-calculation/22041/6
        reconstruction_loss = Losses.RECONSTRUCTION_CRITERION(X, Y)
        if iteration < Losses.WARMUP:
            return reconstruction_loss
        elif iteration == Losses.WARMUP:
            Losses.CENTROIDS = kmeans(X=latent, num_clusters=Losses.N_CLUSTERS)[1].detach()[1].detach()
        k_means_loss = ((latent[:, None] - Losses.CENTROIDS[1]) ** 2).sum(2).min(1)[0].mean()
        loss = reconstruction_loss + alpha * k_means_loss
        return loss
