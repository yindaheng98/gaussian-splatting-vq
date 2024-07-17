import os
from utils.argparse_utils import parser, argument, subcommand
from scene.gaussian_lvq import Attribute, LayeredKMeansGaussianModel

parser.add_argument("--save", type=str, required=True, help="Where to save the layerized codebook.")


@subcommand([
    argument("--save-init", type=str, required=True, help="Where is the codebook."),
    argument("--log2-clusters-init", type=int, required=True, help="Qualtize to how many clusters."),
    argument("--log2-clusters-final", type=int, required=True, help="Merge to how many clusters."),
    argument("--attribute", type=Attribute, choices=list(Attribute), help="Which attribute do you want to quantize."),
    argument("--index", type=int, default=0),
])
def build(args):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    gaussians = LayeredKMeansGaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.init_clusters_path = os.path.join(args.save_init, target_reldir)
    gaussians.build_codebook(args.log2_clusters_init, args.log2_clusters_final, args.attribute, args.index)
    gaussians.save_codebook(os.path.join(args.save, target_reldir), args.attribute, args.index)


@subcommand([
    argument("--dst", type=str, required=True, help="The destination dir."),
    argument("--log2-clusters-init", type=int, default=6, help="Qualtize by how many clusters."),
    argument("--log2-clusters-final", type=int, default=16, help="Qualtize by how many clusters."),
    argument("--log2-clusters-scaling-init", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-scaling-final", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-rotation-init", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-rotation-final", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_dc-init", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_dc-final", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_rest-init", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_rest-final", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-opacity-init", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-opacity-final", type=int, default=0, help="Qualtize by how many clusters."),
])
def quantize(_):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    target_vq_relpath = os.path.join(target_reldir, "point_cloud_vq.ply")
    gaussians = LayeredKMeansGaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.init_clusters_path = os.path.join(args.save, target_reldir)
    gaussians.load_codebooks(
        os.path.join(args.save, target_reldir),
        args.log2_clusters_scaling_init or args.log2_clusters_init,
        args.log2_clusters_scaling_final or args.log2_clusters_final,
        args.log2_clusters_rotation_init or args.log2_clusters_init,
        args.log2_clusters_rotation_final or args.log2_clusters_final,
        args.log2_clusters_features_dc_init or args.log2_clusters_init,
        args.log2_clusters_features_dc_final or args.log2_clusters_final,
        args.log2_clusters_features_rest_init or args.log2_clusters_init,
        args.log2_clusters_features_rest_final or args.log2_clusters_final,
        args.log2_clusters_opacity_init or args.log2_clusters_init,
        args.log2_clusters_opacity_final or args.log2_clusters_final)
    gaussians.test_all()
    gaussians.save_ply(os.path.join(args.dst, target_relpath))
    gaussians.save_vq_ply(os.path.join(args.dst, target_vq_relpath))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)
