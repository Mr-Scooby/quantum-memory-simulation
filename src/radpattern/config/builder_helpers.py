#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from numbers import Real
from typing import Any, Dict

from radpattern.physics import ExpBaseParams
def resolve_theta_max(value: Any, exp: ExpBaseParams) -> float:
    """
    Accepted sim.theta_max:
      - number: explicit angle in radians
      - "forward_lobe_<int>x": <int> * exp.forwardlobe_angular_width
      - "full_sphere": pi
    """

    if isinstance(value, Real) and not isinstance(value, bool):
        theta_max = float(value)

    elif isinstance(value, str):
        key = value.strip().lower()

        if key == "full_sphere":
            theta_max = math.pi

        elif key.startswith("forward_lobe_") and key.endswith("x"):
            factor_txt = key.removeprefix("forward_lobe_").removesuffix("x")

            try:
                factor = int(factor_txt)
            except ValueError:
                raise ValueError(
                    f"Invalid theta_max '{value}'. "
                    "Use format 'forward_lobe_<int>x', for example 'forward_lobe_10x'."
                )

            if factor <= 0:
                raise ValueError("forward_lobe multiplier must be positive.")

            theta_max = factor * exp.forwardlobe_angular_width

        else:
            raise ValueError(
                "Invalid sim.theta_max. Use a number in radians, "
                "'full_sphere', or 'forward_lobe_<int>x'."
            )

    else:
        raise TypeError(
            "sim.theta_max must be a number, 'full_sphere', or 'forward_lobe_<int>x'."
        )

    if not (0.0 < theta_max <= math.pi):
        raise ValueError(f"theta_max must be in (0, pi], got {theta_max}")

    return theta_max
