from .gaussian_kmeans import KMeansGaussianModel, Attribute


class LayeredKMeansGaussianModel(KMeansGaussianModel):
    dirpath = ''

    def build_codebook(self, log2_clusters: int, attr: Attribute, i=0):
        super().load_codebook(self.dirpath, log2_clusters, attr, i)
        kmeans = getattr(self, super().get_name(attr, i))
        data = self.get_data(attr, i).detach()
        quant = super().quantize(attr, i)
        raise NotImplementedError("build_codebook not implemented")
