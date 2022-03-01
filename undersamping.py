"""
umdersampling.py
목적 :
사람별 사진이 들어있는 하위 폴더들로 구성된 데이터셋을 연령에 따른 비율을 비슷하게 만들기 위해
노년층이 아닌 폴더를 임의로 삭제한 새로운 데이터셋의 생성을 목표로 함
근거 : 
1. 대부분의 머신러닝, 딥러닝 모델은 클래스 비율이 비슷할 때 예측 성능이 높아짐
2. 언더샘플링은 최신기술과는 거리가 멀지만 근거1에 부합하는 방식이며 이에 대해 쉽게 이해할 수 있음
3. 오버샘플링의 경우 단순히 minor 클래스 폴더를 복제할 경우 
   a) data leakgae를 피해 train-validation split을 하는 것이 쉽지 않음 
   b) minor 클래스를 복제 후 여러 전처리를 하는 것은 효율적 방법은 단시간에 찾기 어렵고,
      전처리 후 결과가 극적인 경우 같은 클래스로 보기 어려울 가능성이 있고 전처리 후 결과가 비슷하면 leakage 우려가 있음
가정 :
1. 현재 성능 저하의 원인인 노령층에 대한 분류 성능은 나이별 비율이 비슷한 데이터를 이용하면 나아질 것
2. 모델이 쉽게 학습하고 높은 성능을 보이는 청년, 중년층은 기존의 데이터보다 비율이 적어져도 잘 학습할 수 있을 것
사용 방법:
1. 경로는 하드코딩으로 작성되어 어느 위치에 파일을 두어도 작동에 문제는 없음
2. python undersampling.py 실행 후 0~1 사이의 값을 undersampling rate으로 지정해주면 실행됨

Editor : 김대근 (daegunkim0425@gmail.com)
"""

# Import packages
import shutil
import os
import numpy as np
import random

from tqdm import tqdm

# Main script
if __name__=='__main__':
    
    # Fix random seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    path_data_tr_org = '/opt/ml/input/data/train/images'
    path_data_tr_new = '/opt/ml/input/data/train/images'
    
    alpha = float(input('Undersampling rate for not 60+ age groups(Must be 0~1): ')) # Undersampling rate for not 60+ age groups
    if alpha <= 0 or alpha > 1:
        print('Wrong undersampling rate')
        raise NotImplementedError
    elif alpha == 1:
        print('Rate 1 means no undersampling')
        raise NotImplementedError
    
    
    # Copy original data folder
    idx_new = 1
    while True:
        if os.path.isdir(path_data_tr_new + '_' +str(alpha) + '_' + str(idx_new)):
            idx_new += 1
        else:
            path_data_tr_new = path_data_tr_new + '_' +str(alpha) + '_' + str(idx_new)
            shutil.copytree(path_data_tr_org, path_data_tr_new)
            break

        
    # Undersampling by rate alpha
    len_under_60 = 0
    len_60 = 0
    
    for d in tqdm(os.listdir(path_data_tr_new), 'Run undersampling...'):
        if d[0] == '.':
#             print('ignore', d) # ignore useless directory
            continue

        _, gender, race, age = d.split('_')
        
        if int(age) == 60:
            len_60 += 1
            continue
        
        if np.random.uniform() > alpha and int(age) != 60:
            shutil.rmtree(os.path.join(path_data_tr_new, d))
#             print(d, 'is removed')
        else:
            len_under_60 += 1
            
    listdir_org = [x for x in os.listdir(path_data_tr_org) if x[0] != '.']
    listdir_new = [x for x in os.listdir(path_data_tr_new) if x[0] != '.']
            
    print(f'Length of original dir : [{len(listdir_org)}] || new dir : [{len(listdir_new)}]')
    print(f'Length of age 60< : [{len_under_60}] || 60+ : [{len_60}]')
    

