import os
import argparse
from scene import GaussianModel

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source bson.")
parser.add_argument("--dst", type=str, required=True, help="The destination bson.")
parser.add_argument("--log2-clusters", type=int, required=True, help="Qualtize from which layer.")


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


if __name__ == "__main__":
    args = parser.parse_args()
    gaussians = VQGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, "point_cloud.ply"))
    gaussians.VectorQuant()
    os.makedirs(args.dst, exist_ok=True)
    gaussians.save_ply(os.path.join(args.dst, "point_cloud.ply"))
