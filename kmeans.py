import os
import argparse
from scene.gaussian_vq import KMeansGaussianModel, Attribute

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source dir.")
parser.add_argument("--save", type=str, required=True, help="Where to save the codebook.")
parser.add_argument("--iteration", type=int, required=True, help="The source iteration.")

# https://gist.github.com/mivade/384c2c41c3a29c637cb6c603d4197f9f
subparsers = parser.add_subparsers(dest="subcommand")


def argument(*name_or_flags, **kwargs):
    """Convenience function to properly format arguments to pass to the
    subcommand decorator.
    """
    return (list(name_or_flags), kwargs)


def subcommand(args=[], parent=subparsers):
    """Decorator to define a new subcommand in a sanity-preserving way.
    The function will be stored in the ``func`` variable when the parser
    parses arguments so that it can be called directly like so::
        args = cli.parse_args()
        args.func(args)
    Usage example::
        @subcommand([argument("-d", help="Enable debug mode", action="store_true")])
        def subcommand(args):
            print(args)
    Then on the command line::
        $ python cli.py subcommand -d
    """
    def decorator(func):
        parser = parent.add_parser(func.__name__, description=func.__doc__)
        for arg in args:
            parser.add_argument(*arg[0], **arg[1])
        parser.set_defaults(func=func)
    return decorator


@subcommand([
    argument("--log2-clusters", type=int, required=True, help="Qualtize to how many clusters."),
    argument("--attribute", type=Attribute, choices=list(Attribute), help="Which attribute do you want to quantize."),
    argument("--index", type=int, default=0),
])
def build(args):
    args = parser.parse_args()
    target_reldir = os.path.join("point_cloud", f"iteration_{args.iteration}")
    target_relpath = os.path.join(target_reldir, "point_cloud.ply")
    gaussians = KMeansGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.build_codebook(args.attribute, args.log2_clusters, args.index)
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
    gaussians = KMeansGaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(args.src, target_relpath))
    gaussians.load_and_test_all(
        os.path.join(args.save, target_reldir),
        args.log2_clusters_scaling or args.log2_clusters,
        args.log2_clusters_rotation or args.log2_clusters,
        args.log2_clusters_features_dc or args.log2_clusters,
        args.log2_clusters_features_rest or args.log2_clusters,
        args.log2_clusters_opacity or args.log2_clusters)
    gaussians.save_ply(os.path.join(args.dst, target_relpath))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
    else:
        args.func(args)
