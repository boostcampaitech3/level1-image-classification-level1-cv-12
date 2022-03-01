1. NNI 개발 github인 [https://github.com/microsoft/nni](https://github.com/microsoft/nni) 에 따라 python -m pip install --upgrade nni를 실행합니다.    
2. github에 올린 nni폴더와 train.py를 다운받고, nni폴더는 train.py가 존재하는 경로와 동일한 위치로 이동합니다. (ex. train.py의 경로가 code/train.py 라면, nni 폴더는 code/nni)    
3. nni 폴더 내에 존재하는 search_space.json을 원하는 hyperparameter 실험 값으로 입력 합니다.  
4. terminal로 돌아와서 NNI 개발 github와 같이 nnictrl create --config nni/config.yml --port 30001를 입력합니다. (port가 30001은 아니어도 실행은 되는 것으로 알고 있습니다. port를 30001로 설정한 이유는 우리 서버에서 port를 할당할 때 주어진 port가 저는 30001, 30002, 30003, 30004로 나와서 일단 30001로 했습니다.)  
5. 위 명령어를 입력하면 ip:port로 하나의 주소가 뜰 텐데, 이를 무시하고 ai stages에 주어져 있는 서버 ip:port를 주소창에 입력하면 실행 됩니다! (단, experiment의 시간이 오래걸리는 점에 주의하세요ㅠㅠ)  
