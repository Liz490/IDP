"""DataGenerator to handle data loading."""

from keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):

    def __init__(self, X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data,
                 X_methylation_data, Y, batch_size):
        self.drug_feat = X_drug_feat_data
        self.drug_adj = X_drug_adj_data
        self.mutation = X_mutation_data
        self.gexpr = X_gexpr_data
        self.methylation = X_methylation_data
        self.x = [X_drug_feat_data, X_drug_adj_data, X_mutation_data, X_gexpr_data, X_methylation_data]
        self.y = Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [data[idx * self.batch_size:(idx + 1) * self.batch_size] for data in self.x]
        batch_Y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_Y
