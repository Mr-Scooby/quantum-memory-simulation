#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from radpattern.plotting.pattern_3d import plot_pattern_3d
from radpattern.plotting import load_data
from pathlib import Path
import matplotlib.pyplot as plt 

file = Path(input("File: "))
#file = Path("/Users/radek/Documents/universidad/clases/TFM/codes/data/test/Cs133_full_sphere.npz")

data,grid, exp, sim =  load_data(file)

print(data.keys()) 
plot_pattern_3d(grid, data["intensity"][0], title= exp.label )
plt.show()
