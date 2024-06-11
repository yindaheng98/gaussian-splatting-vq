import torch
import pickle
import os
import re
import abc
from sklearn.cluster import MiniBatchKMeans
from .gaussian_model import GaussianModel
from enum import Enum


class Attribute(Enum):
    scaling = 'scaling'
    rotation = 'rotation'
    features_dc = 'features_dc'
    features_rest = 'features_rest'
    opacity = 'opacity'

    def __str__(self):
        return self.value


class VQGaussianModel(GaussianModel, metaclass=abc.ABCMeta):
    method = None

    def get_name(self, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr))
        if data.ndim <= 2:
            name = f"{self.method}_{attr}"
        elif data.ndim == 3:
            if data.shape[1] == 1:
                name = f"{self.method}_{attr}"
            else:
                name = f"{self.method}_{attr}_{i}"
        else:
            raise ValueError("Not supported")
        return name

    def get_data(self, attr: Attribute, i=0):
        data = getattr(self, "_" + str(attr)).detach()
        if data.ndim <= 2:
            kdata = data
        elif data.ndim == 3:
            if data.shape[1] == 1:
                kdata = data[:, 0, ...]
            else:
                kdata = data[:, i, ...]
        else:
            raise ValueError("Not supported")
        return kdata

    @abc.abstractmethod
    def build_codebook(self, attr: Attribute, log2_clusters: int, i=0):
        pass

    def save_codebook(self, dirpath, attr: Attribute, i=0):
        path = os.path.join(dirpath, self.get_name(attr, i) + ".pkl")
        kmeans = getattr(self, self.get_name(attr, i))
        print(f"save codebook {path}.")
        with open(path, "wb") as f:
            pickle.dump(kmeans, f)

    def load_codebook(self, dirpath, attr: Attribute, i=0):
        path = os.path.join(dirpath, self.get_name(attr, i) + ".pkl")
        print(f"load codebook {path}.")
        with open(path, "rb") as f:
            kmeans = pickle.load(f)
            setattr(self, self.get_name(attr, i), kmeans)

    @abc.abstractmethod
    def quantize(self, attr: Attribute, i=0):
        pass

    def set_data(self, attr: Attribute, kdata, i=0):
        data = getattr(self, "_" + str(attr))
        data.requires_grad_(False)
        if data.ndim <= 2:
            data[...] = kdata
        elif data.ndim == 3:
            if data.shape[1] == 1:
                data[:, 0, ...] = kdata
            else:
                data[:, i, ...] = kdata
        else:
            raise ValueError("Not supported")
        data.requires_grad_(True)

    @abc.abstractmethod
    def dequantize(self, attr: Attribute, quant, i=0):
        pass

    def load_and_test(self, dirpath, attr: Attribute, i=0):
        self.load_codebook(dirpath, attr, i)
        self.dequantize(attr, self.quantize(attr, i))

    def parse_name(self, name):
        find = re.findall(rf"{self.method}_([a-z_]+)_([0-9]+)", name)
        if len(find) <= 0:
            find = re.findall(rf"{self.method}_([a-z_]+)", name)
            (attr,) = find
            i = 0
        else:
            attr, i = find[0]
            i = int(i)
        return attr, i

    def load_and_test_all(self, dirpath):
        for entry in os.scandir(dirpath):
            name, ext = os.path.splitext(entry.name)
            if not ext == ".pkl":
                continue
            attr, i = self.parse_name(name)
            self.load_and_test(dirpath, attr, i)


class KMeansGaussianModel(VQGaussianModel):
    method = "kmeans"

    def build_codebook(self, attr: Attribute, log2_clusters: int, i=0):
        kmeans = MiniBatchKMeans(n_clusters=2**log2_clusters, random_state=0, n_init="auto", verbose=1)
        data = self.get_data(attr, i)
        print(f"{log2_clusters} bit Kmeans {self.get_name(attr, i)}. shape: {data.shape}")
        kmeans.fit(data.cpu())
        setattr(self, self.get_name(attr, i), kmeans)

    def quantize(self, attr: Attribute, i=0):
        kmeans = getattr(self, self.get_name(attr, i))
        data = self.get_data(attr, i)
        print(f"quantize by {self.get_name(attr, i)}. shape: {data.shape}")
        return kmeans.predict(data.cpu())

    def dequantize(self, attr: Attribute, quant, i=0):
        kmeans = getattr(self, self.get_name(attr, i))
        data = getattr(self, "_" + str(attr))
        print(f"dequantize by {self.get_name(attr, i)}.")
        self.set_data(attr, torch.tensor(kmeans.cluster_centers_[quant], dtype=data.dtype, device=data.device), i)

    def load_and_test(self, dirpath, attr: Attribute, i=0):
        self.load_codebook(dirpath, attr, i)
        self.dequantize(attr, self.quantize(attr, i))
