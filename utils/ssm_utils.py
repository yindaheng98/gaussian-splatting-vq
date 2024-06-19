import torch


def ssm_1d(seq):
    print(seq)
    return seq


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 1))
    ax = fig.subplots(nrows=2)
    path = "output/vq-coffee_martini/frame1/point_cloud/iteration_30000/kmeans_16_features_dc.npz"
    kmeans = torch.FloatTensor(np.load(path)["codebook"])
    ax[0].imshow(kmeans.unsqueeze(0).expand(1000, -1, -1).cpu().numpy())
    kmeans = torch.FloatTensor(kmeans).to('cuda')
    kmeans = ssm_1d(kmeans)
    ax[1].imshow(kmeans.unsqueeze(0).expand(1000, -1, -1).cpu().numpy())
    ax[0].axis(False)
    ax[1].axis(False)
    plt.show()
