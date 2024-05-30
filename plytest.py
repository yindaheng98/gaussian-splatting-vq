import os
from scene import GaussianModel


class VQGaussianModel(GaussianModel):
    def requires_grad_(self, mode: bool):
        print("xyz", self._xyz.shape)
        print("scaling", self._scaling.shape)
        self._scaling.requires_grad_(mode)
        print("rotation", self._rotation.shape)
        self._rotation.requires_grad_(mode)
        print("features_dc", self._features_dc.shape)
        self._features_dc.requires_grad_(mode)
        print("features_rest", self._features_rest.shape)
        self._features_rest.requires_grad_(mode)
        print("opacity", self._opacity.shape)
        self._opacity.requires_grad_(mode)

    def VectorQuant(self):
        self.requires_grad_(False)
        self._rotation[...] = 0
        self._features_dc[...] = 0
        self.requires_grad_(True)


gaussians = VQGaussianModel(sh_degree=3)
gaussians.load_ply("output/da1fd9e7-c/point_cloud/iteration_30000/point_cloud.ply")
gaussians.VectorQuant()
os.makedirs("output/da1fd9e7-c/point_cloud/iteration_30001", exist_ok=True)
gaussians.save_ply("output/da1fd9e7-c/point_cloud/iteration_30001/point_cloud.ply")
