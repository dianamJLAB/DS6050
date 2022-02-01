##Script for creating Markov chain haplotypes. Very inefficient in its current form.
##Example run: python markover_batch.py 1000G_real_genomes/805_SNP_1000G_real.hapt 50 10 1

import numpy as np
import pandas as pd
import sys

#inpt = "1000G_real_genomes/805_SNP_1000G_real.hapt"
inpt = str(sys.argv[1]) #input file in hapt format. except for the first two columns, each column is a position ("0" or "1"), each row is a haplotype
markov_num = int(sys.argv[2]) #number of markov chain haplotypes to be generated
window_leng = int(sys.argv[3]) #window length
ident = int(sys.argv[4]) #integer identifier to be added at beginning of the output file name (useful for parallel computing)
outpt = str(ident) + "_markov_w" + str(window_leng) + ".hapt" #name of the output file

df1 = pd.read_csv(inpt, sep = ' ', header=None) #read input file
df1 = df1.iloc[:,2:]
df1.columns = list(range(df1.shape[1]))

markovs = []

##Takes a dataframe, calculates frequency of "0" based on the last column for the combined values in the columns except the last one.
##Based on this frequency, "0" or "1" would be picked by probability at a given position.
def mark_calc0(df_sub):
    if len(df_sub.columns) == 2:
        temp = df_sub.iloc[:,0].astype(str)
    else:
        temp = df_sub.iloc[:,0].astype(str).str.cat(df_sub.iloc[:,1:-1].astype(str))
    df_sub["comb"] = temp
    comb_dict0 = {}
    for comb in set(df_sub["comb"]):
        temp_comb = df_sub.loc[df_sub['comb'] == comb]
        comb_dict0[comb] = list(temp_comb.iloc[:,-2].values).count(0)/len(temp_comb)
    return comb_dict0

##Cut a dataframe based on window size
def cutter(df, window_size, i):
    sub_df = df[:,i-window_size:i+1]
    sub_df = pd.DataFrame(sub_df)
    return sub_df

##Function for creating a single markov chain given the dataframe and window length
def markover(df, window_len):
    markov = []
    window_len_temp = window_len
    df = df.values
    for i1 in range(len(df[0,:])):
        ##Put 1 or 0 based solely on frequency for the 1st index
        if i1 == 0:
            prob0 = list(df[:,i1]).count(0)/len(df)
            if prob0 >= np.random.uniform(0, 1):
                markov.append(0)
            else:
                markov.append(1)
            continue
        ##Window length increases from 1 till constant user provided "window_len"
        if i1 <= window_len_temp:
            window_len = i1
        else:
            window_len = window_len_temp
        sub_df = cutter(df, window_len, i1)
        prob0_dict = mark_calc0(sub_df) #frequency of "0" at the position is calculated for the given window
        comb = ''.join(str(e) for e in markov[(i1-window_len):i1])
        if prob0_dict[comb] >= np.random.uniform(0, 1): #decide if "0" or "1" will be put on the position
            markov.append(0)
        else:
            markov.append(1)
    return markov

##Create a data frame of "markov_num" number of markov chain haplotypes
for i in range(markov_num):
    #print(i)
    markovs.append(markover(df1, window_leng))
markovs_df = pd.DataFrame(markovs)

##Create names for each markov chain haplotype
mark_names = []
for i in range(0,len(markovs_df)):
    mark_names.append('Mark'+str(i))

markovs_df.insert(loc=0, column='ID', value=mark_names)
markovs_df.insert(loc=0, column='Type', value="Markov")

##Output file
markovs_df.to_csv(outpt, sep=" ", header=False, index=False)
