"""
compare_two_models
목적 : 두 모델의 validation 예측 결과에 대한 F1-score 시각화 및 비교 분석
사용 주의사항
1. 해당 파일을  train.py와 같은 directory에 위치해야함
2. 파일을 작동시키기 전 비교할 두 모델의 예측 결과가 exp 폴더에 pred_result.csv 형태로 저장되어 있어야함

Editor : Daegun Kim (daegunkim0425@gmail.com)
"""

# Import packages
import numpy as np
import pandas as pd
import os
import itertools as it

from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


# Custom functions
def f1_dict_to_dataframe(report:dict) -> pd.DataFrame:
    df = pd.DataFrame(columns=['precision', 'recall', 'f1', 'count'],
                      index=[int(x) for x in list(report.keys())[:-3]],
                      dtype=np.float32
                      )
    
    for idx in df.index:
        k = str(idx)
        df.loc[idx, :] = report[k]['precision'], report[k]['recall'], report[k]['f1-score'], int(report[k]['support'])
    
    return df


def rearrange_by_feature(df_org:pd.DataFrame):
    mapping_mask = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0,
                    6:1, 7:1, 8:1, 9:1, 10:1, 
                    11:2, 12:2, 13:2, 14:2, 15:2, 16:2, 17:2}
    
    mapping_gender = { 0:0,  1:0,  2:0, 
                       6:0,  7:0,  8:0, 
                      12:0, 13:0, 14:0,
                       3:1,  4:1,  5:1,
                       9:1, 10:1, 11:1,
                      15:1, 16:1, 17:1}
    
    mapping_age = {0:0, 3:0, 6:0,  9:0, 12:0, 15:0,
                   1:1, 4:1, 7:1, 10:1, 13:1, 16:1,
                   2:2, 5:2, 8:2, 11:2, 14:2, 17:2}
    
    mappings = [mapping_mask, mapping_gender, mapping_age]
    
    list_df = []
    for mapping in mappings:
        df = deepcopy(df_org)
        df.true = df.true.map(mapping)
        df.pred = df.pred.map(mapping)
        list_df.append(df)
    
    features = ['mask', 'gender', 'age']
    dict_dfs = {k:df for k, df in zip(features, list_df)}
    
    return dict_dfs # df_mask, df_gender, df_age


# Main script
if __name__=="__main__":
    # Get the indices of two models for comparing
    idx_mdl1 = ''#str(input('First model index[Just enter space for index 1]: ()'))
    idx_mdl2 = '2'#str(input('Second model index[Just enter space for index 1]: ()'))

    path_rslt1 = f'./model/exp{idx_mdl1}/pred_result.csv'
    path_rslt2 = f'./model/exp{idx_mdl2}/pred_result.csv'
    
    df1 = pd.read_csv(path_rslt1)
    df2 = pd.read_csv(path_rslt2)
    
    if idx_mdl1 == '': idx_mdl1 = 1
    if idx_mdl2 == '': idx_mdl2 = 1   
    
    # Get total score by report
    df_f1_tot1 = f1_dict_to_dataframe(classification_report(df1.true, df1.pred, output_dict=True))
    df_f1_tot2 = f1_dict_to_dataframe(classification_report(df2.true, df2.pred, output_dict=True))
    
    print(f'Total macro f1 of the Model [{idx_mdl1}]', df_f1_tot1.f1.mean())
    print(f'Total macro f1 of the Model [{idx_mdl2}]', df_f1_tot2.f1.mean())
    
    # Rearrange results by mask status    
    dict_df1 = rearrange_by_feature(df1)
    dict_df2 = rearrange_by_feature(df2)

    # Make directory to save comparing figures
    try:
        os.mkdir('./compare')
    except FileExistsError:
        pass
    
    try:
        path_figures = f'./compare/compare_{idx_mdl1}_and_{idx_mdl2}'
        os.mkdir(path_figures)
    except FileExistsError as e:
        print(50*'=', 'Warning!: Model comparison directory already exists', 50*'=', sep='\n')
        
    # Get reports from classification report
    dict_reports1 = {k: classification_report(v.true, v.pred, output_dict=True) for k, v in dict_df1.items()}
    dict_reports2 = {k: classification_report(v.true, v.pred, output_dict=True) for k, v in dict_df2.items()}
    
    dict_f1_df1 = {k:f1_dict_to_dataframe(v) for k, v in dict_reports1.items()}
    dict_f1_df2 = {k:f1_dict_to_dataframe(v) for k, v in dict_reports2.items()}
    
    # Plot - total plot 
    fig_tot, ax_tot = plt.subplots(3, 3, figsize=(16, 12))
    
    width_tot = 0.3
    
    for i, left in enumerate(['Mask', 'Gender', 'Age']):
        ax_tot[i][0].set_ylabel(left, fontweight='bold')
    
    for i, top in enumerate(['Precision', 'Recall', 'F1-score']):
        ax_tot[0][i].set_title(top, fontweight='bold')
               
    for i, feature in enumerate(dict_f1_df1.keys()):
        for j, metric in enumerate(dict_f1_df1[feature].columns[:-1]):
            ax_tot[i][j].bar(x=dict_f1_df1[feature][metric].index - width_tot/2,
                             height=dict_f1_df1[feature][metric],
                             width=width_tot,
                             label=f'Model {idx_mdl1}')
            ax_tot[i][j].bar(x=dict_f1_df2[feature][metric].index + width_tot/2,
                             height=dict_f1_df2[feature][metric],
                             width=width_tot,
                             label=f'Model {idx_mdl2}')
            
            min1 = dict_f1_df1[feature][metric].min()
            min2 = dict_f1_df2[feature][metric].min()
            min12 = min(min1, min2)
            ax_tot[i][j].set_ylim([0.9*min12, 1.05])
            ax_tot[i][j].grid(axis='y')
            
    for i, j in it.product(np.arange(0, 3), np.arange(0, 3)):
        if i == 0:
            ax_tot[i][j].set_xticks([0, 1, 2])        
            ax_tot[i][j].set_xticklabels(['Correct', 'Incorrect', 'Not Wear'], fontweight='bold')
        elif i == 1:
            ax_tot[i][j].set_xticks([0, 1])        
            ax_tot[i][j].set_xticklabels(['Male', 'Female'], fontweight='bold')
        else:
            ax_tot[i][j].set_xticks([0, 1, 2])        
            ax_tot[i][j].set_xticklabels(['~29', '30~59', '60+'], fontweight='bold')
        
            
    for i, j in it.product(np.arange(0, 3), np.arange(0, 3)):
        for s in ['top', 'right']:
            ax_tot[i][j].spines[s].set_visible(False)
            ax_tot[i][j].legend()
    
    fig_tot.suptitle(f'Macro F1\nModel {idx_mdl1}: [{df_f1_tot1.f1.mean():.3}] || Model {idx_mdl2}: [{df_f1_tot2.f1.mean():.3}]', fontsize=20, fontweight='bold')
    fig_tot.savefig(path_figures + '/total_viz.png')