"""Plotting and result-loading utilities."""

from .data_loader import load_data
from importlib.resources import files


THESIS_STYLE = str(files(__package__) / "plot_style.mplstyle") 




__all__ = ["load_data"]
