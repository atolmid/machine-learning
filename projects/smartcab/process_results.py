#Reads the results stored in "smartcab.xlsx" and process them
import pandas as pd
import numpy as np
from collections import defaultdict

#open and read the file
df = pd.read_excel('smartcab.xlsx')
df.apply(lambda x: pd.lib.infer_dtype(x.values))

#list to store intermediate results
d = []
#list to store results
results = []
#for the combination of parameters used
for alpha in np.arange(0.0, 1.1, 0.1):
        for gamma in np.arange(0.0, 1.1, 0.1):
            for epsilon in np.arange(0.0, 0.2, 0.05):
            #for each alpha, gamma and epsilon combination gather and process the reaults
                df_filtered = df[(df.alpha == alpha) & (df.gamma == gamma) & (df.epsilon == epsilon)]
                #summarize the number of successes
                success = np.sum(df_filtered['successes %'])
                #
                m = []
                mov = []
                #number of times the last 10 attempts were successful
                last  = np.sum(df_filtered['Last 10 Runs Successful'] == True)
                #convert the data to the appropriate format, then estimate the average rewards per step
                for i in df_filtered['rewards per step']:
                    d = i.encode('utf-8')
                    for j in d:
                        j.encode('UTF8')
                    k = d.split(",")
                    k[0] = k[0][1:]
                    k[len(k)-1] = k[len(k)-1].split("]")[0]
                    l = np.array(k).astype(np.float)
                    m.append(np.mean(l))
                #find the average for all runs
                smean = np.mean(m)
                #convert the data to the appropriate format, then estimate the average number of moves
                for i in df_filtered['number of moves']:
                    d = i.encode('utf-8')
                    for j in d:
                        j.encode('UTF8')
                    k = d.split(",")
                    k[0] = k[0][1:]
                    k[len(k)-1] = k[len(k)-1].split("]")[0]
                    l = np.array(k).astype(np.float)
                    mov.append(np.mean(l))
                #find the average for all runs
                avg_moves = np.mean(mov)
                #append calculation results to the results list for all parameter combinations
                results.append([alpha, gamma, epsilon, last, success, smean, avg_moves])
#export the results to an excel file
df_results = pd.DataFrame(results, columns = ['alpha', 'gamma', 'epsilon', 'Last 10 Runs Successful', 'successes','average rewards per step', 'average number of moves'])
df_results.to_excel('Processed_Results.xlsx')