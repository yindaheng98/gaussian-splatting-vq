import os
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans as KMeans
from .gaussian_vq import VQGaussianModel, Attribute


class KMeansGaussianModel(VQGaussianModel):

    def kmeans_filename(self, log2_clusters: int, attr: Attribute, i=0):
        return self._get_filename("kmeans", log2_clusters, attr, i)

    def kmeans_varname(self, attr: Attribute, i=0):
        return self._get_name("kmeans", attr, i)

    def kmeans_log2_clusters_varname(self, attr: Attribute, i=0):
        return f"log2_clusters_{self.kmeans_varname(attr, i)}"

    @staticmethod
    def kmeans_batch(log2_clusters, datasize):
        return datasize // 10

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        data = self.get_data(attr, i).detach()
        kmeans = KMeans(n_clusters=2**log2_clusters, init='random', random_state=0,
                        n_init="auto", verbose=1, batch_size=self.kmeans_batch(log2_clusters, data.shape[0]))
        print(f"{log2_clusters} bit Kmeans {self.kmeans_varname(attr, i)}. shape: {data.shape}")
        kmeans.fit(data.cpu())
        setattr(self, self.kmeans_varname(attr, i), torch.FloatTensor(kmeans.cluster_centers_).to(data.device))
        setattr(self, self.kmeans_log2_clusters_varname(attr, i), log2_clusters)

    def save_codebook(self, dirpath, attr: Attribute, i=0):
        os.makedirs(dirpath, exist_ok=True)
        log2_clusters = getattr(self, self.kmeans_log2_clusters_varname(attr, i))
        path = os.path.join(dirpath, self.kmeans_filename(log2_clusters, attr, i) + ".npz")
        kmeans = getattr(self, self.kmeans_varname(attr, i))
        print(f"save codebook {path}.")
        np.savez(path, codebook=kmeans.cpu().numpy())

    def load_codebook(self, dirpath, log2_clusters: int, attr: Attribute, i=0):
        path = os.path.join(dirpath, self.kmeans_filename(log2_clusters, attr, i) + ".npz")
        data = self.get_data(attr, i)
        print(f"load codebook {path}.")
        kmeans = torch.FloatTensor(np.load(path)["codebook"]).to(data.device)
        setattr(self, self.kmeans_varname(attr, i), kmeans)

    quant_batch = 4096

    def quantize(self, attr: Attribute, i=0):
        kmeans = getattr(self, self.kmeans_varname(attr, i))
        data = self.get_data(attr, i).detach()
        print(f"quantize by {self.kmeans_varname(attr, i)}. shape: {data.shape}")
        quantized = torch.zeros(data.shape[0], dtype=torch.int32, device=data.device)
        for i in range(0, data.shape[0], self.quant_batch):
            step = self.quant_batch if i+self.quant_batch < data.shape[0] else i+self.quant_batch - data.shape[0]
            dist = torch.norm(data[i:i+step, ...].unsqueeze(1) - kmeans.unsqueeze(0), p=2, dim=2)
            quantized[i:i+step] = dist.argmin(dim=1)
        return quantized

    def dequantize(self, attr: Attribute, quant, i=0):
        kmeans = getattr(self, self.kmeans_varname(attr, i))
        data = self.get_data(attr, i).detach()
        dequantized = kmeans[quant]
        mean = torch.abs(data).mean(dim=0).cpu().numpy()
        mean_dequantized = torch.abs(dequantized).mean(dim=0).cpu().numpy()
        loss = torch.abs(dequantized - data).mean(dim=0).cpu().numpy()
        self.set_data(attr, dequantized, i)
        print(f"dequantized by {self.kmeans_varname(attr, i)}.")
        print(f"dequantized loss:       {loss}")
        print(f"dequantized rel loss:   {loss / mean_dequantized}")
        print(f"dequantized mean:       {mean_dequantized}")
        print(f"dequantize  mean shift: {mean - mean_dequantized}")
