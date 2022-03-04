# Image Classification
### **1-1. 프로젝트 개요**

- 프로젝트 주제
    
    마스크 착용 상태 분류 : 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task
    
- 프로젝트 개요
    
    카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 모델 구축
    
- 활용 장비 및 재료
    
    서버 : AI Stages / GPU : V100 (AI Stages Server 자체 탑재) / 노트북 : 개인 노트북
    
- 프로젝트 구조 및 사용 데이터셋의 구조도
    
    베이스라인 코드 : 김태진 마스터님과 조교분들이 제공
    
    프로젝트 구조 : 베이스라인 코드의 구조를 따라감
    
- 기대 효과
    
    공중 보건의 인적자원 소모가 큰 공공장소 내 마스크 착용 확인은 CV를 이용해 대체함으로써 감염병 전파 상황에서의 인적자원 효율 증대
    
- 문제 정의
    
    상반신 촬영 컬러 이미지를 이용한 다중 분류 (Multiclass classification) 문제
    

### **1-2. 프로젝트 팀 구성 및 역할**

- 김대근 : 데이터 시각화, 에러 분석. 전략 분석
- 박선혁 : 코드 통합, Hyperparameter 기법 적용 및 실험
- 심준교 : 베이스라인 코드 작성 및 주어진 베이스라인 코드 수정
- 전경민 : Data Augmentation 실험 및 다양한 기법 실험, auto-ml 코드 작성
- 최용원 : 베이스라인을 기반으로 성능을 높이기 위한 Data augmentation에 대한 여러가지 기법 적용 및 실험

### **1-3. 프로젝트 수행 절차 및 방법**

- 기간별 활동 내용
    
    2/21 ~ 2/23 : 개인별 EDA 및 개별 베이스라인 작성
    
    2/24 ~ 2/27 : 주어진 베이스라인 코드 수정 및 모델 선정
    
    2/28 ~ 3/3 : 각종 실험 수행 및 결과 분석
    
- 프로젝트 수행 및 실험 절차의 큰 틀 ***(Fig 1)***
- 협업 방식 : Notion ([https://seoulsky-field.notion.site/Image-Classification-0229a68200a44ced8d3783f4363e6570](https://www.notion.so/Image-Classification-0229a68200a44ced8d3783f4363e6570))

### **1-4. 프로젝트 수행 결과**

1. 탐색적분석 및 전처리 (EDA) - 학습데이터 소개
2. 주요 모델 - EfficientNet
    1. 2019년 ICML에서 소개된 논문 ‘EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks’ 의 아이디어를 기반으로 구현된 모델 ( [https://arxiv.org/pdf/1905.11946.pdf](https://arxiv.org/pdf/1905.11946.pdf) )
    2. CNN 모델의 depth, width, resolution을 기반으로하여 모델의 복잡도를 수치화하여 효율적으로 조절할 수 있는 compound scaling을 제안함 ***(Fig 2)***
    
3. 모델 선정 및 분석
    - 아키텍쳐 : EfficientNet B0
        1. training time augmentation
            
            Resize(224,224,p=1.0), HorizontalFlip(p=0.5), ShiftScaleRotate(p=0.5,rotate_limit=15), HueSaturationValue(hue_shift_limit=0.2,sat_shift_limit=0.2,val_shift_limit=0.2,p=0.5), RandomBrightnessContrast(brightness_limit=(-0.1,0.1) ,contrast_limit=(-0.1,0.1),p=0.5),  A.GaussNoise(p=0.5), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0), ToTensorV2(p=1.0)
            
        2. input size : [224,224]
        3. optimizer : Adam
        4. scheduler : StepLR
        5. 추가 시도
            
            랜덤 시드 변경 -> 학습 및 추론 -> hard-voting을 통해 최종 output 산출
            
            high 제출 score 결과 파일들을 가지고 hard-voting을 통해 최종 output 산출
            
        
4. 모델 평가 및 개선
    - Class 별 모델 취약점 분석 ***(Fig 3)***
    1. 나이가 60세 이상인 class인 2, 5, 8, 11, 14, 17번 class의 F1-score가 특히 낮은 것을 볼 수 있음
    2. 기본 세팅 외 적용했었던 여러 방법론에서도 위 그래프와 같이 60대 이상의 class를 판별하는 것에 특히 어려움을 겪음
    
5. 제출 결과 ***(Fig 4)***
    1. Private Score 기준 F1 Score 0.7316/ Accuracy 78.6349로 전체 48팀 중 28위를 달성함
    2. Public Score 순위에 비해 Private Score 순위가 7등 상승한 것을 통해 ensemble 방식을 적용한 최종 제출 결과물이 좀 더 강건한 결과임을 짐작해 볼 수 있었음
    

### **1-5. 자체 평가 의견**

- 잘한 점들
    1. Git과 Notion등 협업 툴을 시도 및 팀 내 작업에 대해 공유하고 피드백에 노력함
    2. 수업에서 언급되었던 Cutmix, NNI와 같은 방법론을 실험을 통해 적용해봄
    3. 개인 프로젝트와 다르게 팀원들과 이해를 공유하기 위한 노력을 들인 것
    4. 방법론들을 그냥 쓰는 것이 아닌 깊이 있는 이해를 한 후 적용하였다는 점
    
- 시도했으나 잘 되지 않았던 것들
    1. validation set을 뽑을 때 샘플링 기법을 적절하게 사용하지 못한 점
    2. Data Augmentation 관련하여 다양한 실험을 진행하였으나 성능과 직결되지 못함
    3. Semi-Supervised Learning를 적용하려고 마지막에 시도하였으나 실패함
    4. Over-Sampling과 Under-Sampling을 모두 시도하였으나 성능과 직결되지 못함
    
- 아쉬웠던 점들
    1. WandB 등을 통해서 실험 환경 관리를 체계적으로 했으면 좋았을 것
    2. 실력이 부족하여 실험해 보고 싶은 것들을 제대로 성공해보지 못한 것
    3. 실험 전, 코드 작성을 꼼꼼히 하여 이후의 실험들이 무의미한 결과가 되지 않게 하는 것이 효율적이었을 것
    4. 개인 f1 score를 더 올려보고 싶었으나 성능 향상을 많이는 못한 것
    
- 프로젝트를 통해 배운 점 또는 시사점
    1. 높은 Class imbalance에 대응하는 다양한 방식들에 대해 알게 됨
    2. 데이터의 양도 많고 품질도 좋은 것으로 학습을 시키는데는 이유가 있다는 점
    3. 최신 기술이나 좋다고 알려진 것들을 무작정 따라가는 것 보다, 현재 진행하고 있는 프로젝트에 적합한 기술을 접목하는게 중요하다고 느낌
    4. 실험 버전 관리 및 체계적인 실험 계획이 중요함을 깨달음
    5. 실제 프로젝트를 진행하면서 “baseline”을 구축하는 것이 가장 어렵다는 생각이 듦
    

### **Appendix. Figures**

- Fig. 1 Project flow chart

![image](https://user-images.githubusercontent.com/83350060/156701037-b876602b-430a-49ff-90f0-3c379dc838c7.png)



- Fig. 2 Schema of compound scaling

![image](https://user-images.githubusercontent.com/83350060/156701381-49ba469d-0544-4e0d-bd1c-4f53aa72989d.png)



- Fig. 3 Validation evaluation results by each classes for the basic model in the team
    (Left : Precision and Recall score  /  Right : F1-score)

![image](https://user-images.githubusercontent.com/83350060/156701457-807ead67-184e-4852-a807-1ecc50f2143c.png)


    
- Fig. 4 Final Submission result

![image](https://user-images.githubusercontent.com/83350060/156701509-5905544c-079e-4fcf-8be5-9eada38e04ca.png)

