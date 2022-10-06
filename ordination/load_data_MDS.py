# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:34:35 2022

@author: basvo
"""
import os
import pandas as pd
import numpy as np
from sklearn.manifold import MDS

tax_table = pd.read_csv('cirrhosis_taxtable.txt',delimiter='\t')

tax_tree = pd.DataFrame(tax_table['OTU ID'])
tax_tree = tax_tree['OTU ID'].str.split(';', expand=True)

#remobve all rows that have a None value in the Species column
index_of_interest = tax_tree.iloc[:,6].isnull() == False
tax_tree = tax_tree.loc[index_of_interest,:]
tax_table = tax_table.loc[index_of_interest,:]

#keep only the rows that have None value in Taxa column
index_of_interest = tax_tree.iloc[:,7].isnull()
tax_tree = tax_tree.loc[index_of_interest,:6]
tax_table = tax_table.loc[index_of_interest,:]
tax_table['OTU ID'] = tax_tree.iloc[:,6]

index_of_interest = tax_table != 0
index_of_interest = np.sum(index_of_interest,axis=1) == 0

#save as numpy array and initialize the distance matrix
df_tax = tax_tree.to_numpy()
distmat = np.zeros((df_tax.shape[0],df_tax.shape[0]))



print('- BUILDING DISTANCE MATRIX -')
count=0

for i in range(df_tax.shape[0]):
    
    row1 = df_tax[i,:]
    
    for j in range(count):
        distmat[i,j]=1
        
        row2 = df_tax[j,:]
        
        dist=1
        checkval=0
        for level in reversed(range(6)):
            
            if row1[level]==row2[level]:
                distmat[i,j]=np.log2(dist)+1
                continue
            dist+=1
            distmat[i,j]=np.log2(dist)+1
        
    count+=1
  
distmat = distmat+np.transpose(distmat)

names = tax_tree[6].tolist()
dist_df = pd.DataFrame(distmat, columns = names, index = names)

# Start gathering positions from distances
print('- STARTING MDS FIT -')

mds = MDS(n_components = 2,dissimilarity='precomputed',metric=True,verbose=2,n_init=1,max_iter = 10000,random_state=64)
positions = mds.fit_transform(distmat)

locations = pd.DataFrame(positions,columns=['X','Y'], index = names)
locations.to_csv('cirrhosis_mapping.csv')