import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True, help="The source dir.")
parser.add_argument("--save", type=str, required=True, help="Where to save the codebook.")
parser.add_argument("--iteration", type=int, required=True, help="The source iteration.")
parser.add_argument("--sh-degree", type=int, default=3, help="SH degree.")

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
