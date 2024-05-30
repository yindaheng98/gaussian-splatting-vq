import os
from scene import GaussianModel

class VQGaussianModel(GaussianModel):
    def VectorQuant(self):
        print("scaling", self.get_scaling.shape)
        print("rotation", self.get_rotation.shape)
        print("xyz", self.get_xyz.shape)
        print("features", self.get_features.shape)
        print("opacity", self.get_opacity.shape)
        

gaussians = VQGaussianModel(sh_degree=3)
gaussians.load_ply("output/da1fd9e7-c/point_cloud/iteration_30000/point_cloud.ply")
gaussians.VectorQuant()
os.makedirs("output/da1fd9e7-c/point_cloud/iteration_30001", exist_ok=True)
gaussians.save_ply("output/da1fd9e7-c/point_cloud/iteration_30001/point_cloud.ply")
