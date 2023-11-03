"""Script to calculate mean and standard deviation"""

from statistics import stdev, mean

cell_out = [0.868, 0.873, 0.857, 0.851, 0.859]
drug_out = [0.455, 0.362, 0.222, 0.251, 0.541]
both_out = [0.3155, 0.0168, 0.1177, 0.2607, 0.19037]
mut = [0.916, 0.926, 0.923, 0.926, 0.923]
gexpr = [0.921, 0.922, 0.919, 0.918, 0.925]
methy = [0.527, 0.516, 0.499, 0.560, 0.333]
drug_random = [0.264, 0.177, 0.436, 0.252, 0.192]
normal = [0.925, 0.925, 0.922, 0.926, 0.923]
normal_10 = []
normal_20 = [0.045, 0.758, 0.4230, 0.507, 0.268]

for setting in [methy]:
    print(mean(setting))
    print(stdev(setting))
