import pandas as pd

"""
This script produces a small subset of the data for debugging. 
Using a subset, loading the data is much faster. 
"""

# number of rows loaded from each of the csv files
nb_rows = 50
DPATH = '../data'
Drug_feature_file = '%s/GDSC/drug_graph_feat' % DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv' % DPATH

Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv' % DPATH
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv' % DPATH
Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv' % DPATH

files = {"Genomic_mutation_file": Genomic_mutation_file, "Gene_expression_file": Gene_expression_file,
         "Methylation_file": Methylation_file}

for file in files.keys():
    df = pd.read_csv(files[file], index_col=0)
    df_sorted = df.sort_index()
    df_sorted.head(nb_rows).to_csv(f'{DPATH}/CCLE/Sorted/{file}_sorted.csv')
