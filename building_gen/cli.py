"""Command line entry points for the HOT transfer experiments."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "train-baseline", "evaluate-baseline"])
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "benchmark":
        from building_gen.benchmark.cli import main as run
    elif args.command == "train-baseline":
        from building_gen.wrapper.rl_cli import main as run
    else:
        from building_gen.wrapper.rl_cli import evaluate_main as run
    return run(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
