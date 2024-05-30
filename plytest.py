import os
from scene import GaussianModel

class VQGaussianModel(GaussianModel):
    def VectorQuant(self):
        print("scaling", self._scaling.shape)
        print("rotation", self._rotation.shape)
        print("xyz", self._xyz.shape)
        print("features_dc", self._features_dc.shape)
        print("features_rest", self._features_rest.shape)
        print("opacity", self._opacity.shape)
        

gaussians = VQGaussianModel(sh_degree=3)
gaussians.load_ply("output/da1fd9e7-c/point_cloud/iteration_30000/point_cloud.ply")
gaussians.VectorQuant()
os.makedirs("output/da1fd9e7-c/point_cloud/iteration_30001", exist_ok=True)
gaussians.save_ply("output/da1fd9e7-c/point_cloud/iteration_30001/point_cloud.ply")
