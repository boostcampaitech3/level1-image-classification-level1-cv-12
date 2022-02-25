# Import packages
import pandas as pd
import matplotlib as mpl
import os
import numpy as np

from matplotlib import pyplot as plt


# Run main code
if __name__=='__main__':
    
    num_exp = str(input('Exp num: '))
    path_f1_rslt = f'./model/exp{num_exp}/f1_result.csv'
    
    # Change working directory
    os.chdir(f'./model/exp{num_exp}')
    
    # Load f1 result dataframe
    df_f1 = pd.read_csv('f1_result.csv')
    
    # Make f1 analysis figure directory
    try:
        os.mkdir('figures_f1_result')
        os.chdir('figures_f1_result')
    except:
        os.chdir('figures_f1_result')
    
    # Plot - Basic plot of precision, recall, f1 score       
    fig_basic, ax_basic = plt.subplots(1, 1, figsize=(14, 10))

    idx_basic = 3*np.arange(len(df_f1.index))
    width_basic = 0.5
    ax_basic.bar(idx_basic-1*width_basic,
                 df_f1.precision,
                 width=width_basic,
                 label='Precision')
    ax_basic.bar(idx_basic,
                 df_f1.recall,
                 width=width_basic,
                 label='Recall')
    ax_basic.bar(idx_basic+1*width_basic,
                 df_f1.precision,
                 width=width_basic,
                 label='F1')

    ax_basic.set_xticks(idx_basic)
    ax_basic.set_xticklabels([str(x) for x in np.arange(len(df_f1.index))])
    ax_basic.legend(fontsize=15)
    ax_basic.set_ylim(0.9*df_f1.min().min(), 1.1)
    ax_basic.axhline(1, linestyle='--', linewidth=2, color='blue')
#     ax_basic.grid(axis='y')
    
    for s in ['top', 'right']:
        ax_basic.spines[s].set_visible(False)
    
    fig_basic.suptitle('Basic bar chart', fontsize=20)
    fig_basic.savefig('Basic bar chart')
    
    # Plot - Precision and Recall
    fig_pr, ax_pr = plt.subplots(1, 2, figsize=(14, 8))
    
    idx_pr = 3*np.arange(len(df_f1.index))
    width_pr = 0.7
    
    ax_pr[0].bar(idx_pr - width_pr/2,
                 df_f1.precision,
                 width=width_pr,
                 label='Precision')
    ax_pr[0].bar(idx_pr + width_pr/2,
                 df_f1.recall,
                 width=width_pr,
                 label='Recall')
    ax_pr[0].bar(3*df_f1.precision.nsmallest().index[0] - width_pr/2, df_f1.precision.min(),
                 color='green',
                 width=width_pr,
                 label=f'Min P {df_f1.precision.min():.2}')
    ax_pr[0].bar(3*df_f1.recall.nsmallest().index[0] + width_pr/2, df_f1.recall.min(),
                 color='red',
                 width=width_pr,
                 label=f'Min R {df_f1.recall.min():.2}')
    
    ax_pr[0].set_xticks(idx_pr)
    ax_pr[0].set_xticklabels([str(x) for x in np.arange(len(df_f1.index))])
    ax_pr[0].legend(fontsize=10)
    ax_pr[0].set_ylim(0.9*df_f1.min().min(), 1.1)
    ax_pr[0].axhline(1, linestyle='--', linewidth=2, color='blue')  
#     ax_pr[0].grid(axis='y')
    
    ax_pr[1].bar(idx_pr,
                 df_f1.f1,
                 width=2*width_pr,
                 label='F1-score')
    ax_pr[1].bar(3*df_f1.f1.nlargest().index[0], df_f1.f1.max(),
                 color='orange',
                 width=2*width_pr,
                 label=f'Max F1 {df_f1.f1.max():.2}')
    ax_pr[1].bar(3*df_f1.f1.nsmallest().index[0], df_f1.f1.min(),
                 color='red',
                 width=2*width_pr,
                 label=f'Min F1 {df_f1.f1.min():.2}')
    
    ax_pr[1].set_xticks(idx_pr)
    ax_pr[1].set_xticklabels([str(x) for x in np.arange(len(df_f1.index))])
    ax_pr[1].legend(fontsize=10)
    ax_pr[1].set_ylim(0.9*df_f1.min().min(), 1.1)
    ax_pr[1].axhline(1, linestyle='--', linewidth=2, color='blue')  
#     ax_pr[0].grid(axis='y')
    
    for s in ['top', 'right']:
        ax_pr[0].spines[s].set_visible(False)
        ax_pr[1].spines[s].set_visible(False)
    
    fig_pr.suptitle('Two slide bar', fontsize=20)
    fig_pr.savefig('Two slide bar')