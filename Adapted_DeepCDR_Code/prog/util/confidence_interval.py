"""Script to calculate 95%-CI for PCC."""

import numpy as np

pcc_normal = [0.8992, 0.8932, 0.8892, 0.9032, 0.8631, 0.9011, 0.8948, 0.8966, 0.8925, 0.8909, 0.8631, 0.8975, 0.8974,
              0.8887, 0.8842, 0.8924, 0.8689, 0.8963, 0.9149]
pcc_drug_out = [0.4311, 0.3815, 0.4663, 0.4699, 0.4640, 0.4935, 0.3963]
pcc_cell_out = [0.7729, 0.7960, 0.7670, 0.8107, 0.7785, 0.7994, 0.8291]
pcc_all_out = [0.2914, 0.2990, 0.2342, 0.2598, 0.2509, 0.2943, 0.1489]
pcc_all_out_strict = [-0.0158, -0.2329, -0.2602, -0.0913, -0.0526, -0.1520, -0.1494]

pcc_norm = [0.925, 0.925, 0.922, 0.926, 0.923]
# pcc_norm_10 =
pcc_norm_20 = [0.045, 0.758, 0.4230, 0.507, 0.268]

all_n = 88
drug_n = 8095
cell_n = 5588
normal_n = 5372

normal = (pcc_normal, normal_n)
all_strict = (pcc_all_out_strict, all_n)
all = (pcc_all_out, all_n)
cell = (pcc_cell_out, cell_n)
drug = (pcc_drug_out, drug_n)
pcc_norm = (pcc_norm, normal_n)
pcc_norm20 = (pcc_norm_20, normal_n)

if __name__ == '__main__':
    z_alpha = 1.956
    pcc_list, n = pcc_norm20
    for pcc in pcc_list:
        z_r = 0.5 * np.log((1 + pcc) / (1 - pcc))
        z_u = z_r + z_alpha * np.sqrt(1 / (n - 3))
        z_l = z_r - z_alpha * np.sqrt(1 / (n - 3))
        r_l = (np.exp(2 * z_l) - 1) / (np.exp(2 * z_l) + 1)
        r_u = (np.exp(2 * z_u) - 1) / (np.exp(2 * z_u) + 1)
        print(f"[{round(r_l, 4)}, {round(r_u, 4)}]")
