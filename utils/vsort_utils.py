import torch
import json


def vsort_by_tree(kmeans, tree):
    return kmeans


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 1))
    ax = fig.subplots(nrows=2)
    path = "output/vq-coffee_martini/frame1/point_cloud/iteration_30000/lkmeans_12_features_dc.npz"
    tree_path = "output/vq-coffee_martini/frame1/point_cloud/iteration_30000/lkmeans_12_features_dc.json"
    lkmeans = torch.FloatTensor(np.load(path)["codebook"])
    with open(tree_path, "r") as f:
        tree = json.load(f)
    kmeans = lkmeans[0:lkmeans.shape[0]-len(tree), ...]
    ax[0].imshow(kmeans.unsqueeze(0).expand(1000, -1, -1).cpu().numpy())
    kmeans = kmeans.to('cuda')
    kmeans = vsort_by_tree(kmeans, tree)
    ax[1].imshow(kmeans.unsqueeze(0).expand(1000, -1, -1).cpu().numpy())
    ax[0].axis(False)
    ax[1].axis(False)
    plt.show()
