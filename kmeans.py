import os
from utils.argparse_utils import parser, argument, subcommand
from scene.gaussian_kmeans import Attribute, KMeansGaussianModel

parser.add_argument("--save", type=str, required=True, help="Where to save the codebook.")


@subcommand([
    argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters."),
    argument("--attribute", type=Attribute, choices=list(Attribute), help="Which attribute do you want to quantize."),
    argument("--index", type=int, default=0),
])
def build(args):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    gaussians = KMeansGaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.build_codebook(args.log2_clusters, args.attribute, args.index)
    gaussians.save_codebook(os.path.join(args.save, target_reldir), args.attribute, args.index)


@subcommand([
    argument("--dst", type=str, required=True, help="The destination dir."),
    argument("--log2-clusters", type=int, default=16, help="Qualtize by how many clusters."),
    argument("--log2-clusters-scaling", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-rotation", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_dc", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_rest", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-opacity", type=int, default=0, help="Qualtize by how many clusters.")
])
def quantize(_):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    target_vq_relpath = os.path.join(target_reldir, "point_cloud_vq.ply")
    gaussians = KMeansGaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.load_codebooks(
        os.path.join(args.save, target_reldir),
        args.log2_clusters_scaling or args.log2_clusters,
        args.log2_clusters_rotation or args.log2_clusters,
        args.log2_clusters_features_dc or args.log2_clusters,
        args.log2_clusters_features_rest or args.log2_clusters,
        args.log2_clusters_opacity or args.log2_clusters)
    gaussians.test_all()
    gaussians.save_ply(os.path.join(args.dst, target_relpath))
    gaussians.save_vq_ply(os.path.join(args.dst, target_vq_relpath))


@subcommand([
    argument("--dst", type=str, required=True, help="The destination dir."),
    argument("--log2-clusters", type=int, default=16, help="Qualtize by how many clusters."),
    argument("--log2-clusters-scaling", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-rotation", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_dc", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-features_rest", type=int, default=0, help="Qualtize by how many clusters."),
    argument("--log2-clusters-opacity", type=int, default=0, help="Qualtize by how many clusters.")
])
def dequantize(_):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    target_vq_relpath = os.path.join(target_reldir, "point_cloud_vq_ddrc.ply")
    gaussians = KMeansGaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.load_codebooks(
        os.path.join(args.save, target_reldir),
        args.log2_clusters_scaling or args.log2_clusters,
        args.log2_clusters_rotation or args.log2_clusters,
        args.log2_clusters_features_dc or args.log2_clusters,
        args.log2_clusters_features_rest or args.log2_clusters,
        args.log2_clusters_opacity or args.log2_clusters)
    gaussians.load_vq_ply(os.path.join(args.dst, target_vq_relpath))
    gaussians.save_ply(os.path.join(args.dst, target_relpath))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)
