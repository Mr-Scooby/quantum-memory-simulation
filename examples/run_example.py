#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal Cs-133 warm-vapour simulation example."""

from pathlib import Path

from radpattern.config.builder import build_run_objects
from radpattern.simulation.runner import run_one_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/example_cs133.json"
OUTPUT = ROOT / "results"


def main():
    objs = build_run_objects(CONFIG)

    run_one_config(
        objs,
        output_dir=OUTPUT,
        save_full_mc=False,
    )


if __name__ == "__main__":
    main()
