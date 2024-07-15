import os
from utils.argparse_utils import parser, argument, subcommand
from scene.gaussian_kmeans import Attribute
from scene.gaussian_lvq import LayeredKMeansGaussianModel

parser.add_argument("--lsave", type=str, required=True, help="Where to save the layerized codebook.")

@subcommand([
    argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters."),
    argument("--attribute", type=Attribute, choices=list(Attribute), help="Which attribute do you want to quantize."),
    argument("--index", type=int, default=0),
])
def build(args):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    gaussians = LayeredKMeansGaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.dirpath = os.path.join(args.save, target_reldir)
    gaussians.build_codebook(args.log2_clusters, args.attribute, args.index)
    gaussians.save_codebook(os.path.join(args.lsave, target_reldir), args.attribute, args.index)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)
