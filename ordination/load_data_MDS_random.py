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

positions = np.zeros((df_tax.shape[0],2))

pixels_x = 449
pixels_y = 448
for i in range(df_tax.shape[0]):
    
    cur_X = np.random.randint(0,pixels_x)
    cur_Y = np.random.randint(0,pixels_y)
    
    count = 0
    while np.sum(np.sum(np.c_[positions[:,0] == cur_X,positions[:,1] == cur_Y],axis=1)==2)>0:
        cur_X = np.random.randint(0,pixels_x)
        cur_Y = np.random.randint(0,pixels_y)
        
        count+=1

    positions[i,:] = [cur_X,cur_Y]

locations = pd.DataFrame(positions,columns=['X','Y'], index = tax_table['OTU ID'])
locations.to_csv('cirrhosis_mapping_random.csv')
