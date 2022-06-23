# Training
아래 코드를 실행

    python3 train.py

학습이 완료되면 가중치 파일이 config.yaml에서 지정된 경로에 저장됨(TODO 경로 확인 코드 추가)  
save_pth: 가중치 파일을 저장할 경로  
regressor_filename: 가중치 파일 이름  

# submission
아래 코드를 실행

    python3 test.py

test.py와 같은 경로에 submission.csv가 생김

# How to work

### feature engineering
#### dataframe객체 내에서 처리 가능한 간단한 함수
1. config.yaml의 feature_engineering 리스트에 **전처리 함수명**을 **문자열**로 추가
2. model/feature_engineering.py파일에 전처리 함수를 추가

#### dataframe객체 밖의 외부 요소를 인자로 넘겨야 되는 경우
1. model/feature_engineering.py파일에 전처리 함수를 추가
2. feature_engineering 함수에서 원하는 위치에 전처리 함수를 호출

### model configuration
1. config.yaml의 regressor.type에 원하는 이름을 입력
2. model/builder.py의 build_regressor에서 해당 모델을 반환

### handling data
모든 데이터는 data 폴더에 넣어둘것