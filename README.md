# Episode_1_blending_model

## data load
```
# kaggle data load
train = pd.read_csv('/content/drive/MyDrive/cloud_share/머신러닝/2차팀플_data/train_module.csv', index_col='id')
test = pd.read_csv('/content/drive/MyDrive/cloud_share/머신러닝/2차팀플_data/test_module.csv', index_col='id')
submission = pd.read_csv('/content/drive/MyDrive/cloud_share/머신러닝/2차팀플_data/sample_submission_module.csv')
```
### 추가 데이터 로드
```
### 추가 데이터 로드
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
original_df = pd.DataFrame(housing.data, columns=housing.feature_names)
original_df[housing.target_names[0]] = housing.target
```
- 원본 train data 37137개 + 추가 train data 20640개 총 57777개의 train data 사용
- test data 24759개

--------------------------------------------------------
![feature_ex](https://user-images.githubusercontent.com/39111185/218971428-e54d73a0-6083-41e2-9039-8976b363fa53.png)
- 독립변수 8개, 종속변수 1개

--------------------------------------------------------
### feature 설명
![feature](https://user-images.githubusercontent.com/39111185/218971755-fdcc7c47-78e9-4463-8040-a2ebfcee7d97.png)

--------------------------------------------------------

## EDA
### library load
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from umap import UMAP

%matplotlib inline
```

### pandas_profiling_report
```
from pandas_profiling import ProfileReport
profile = ProfileReport(train)
profile.to_notebook_iframe()
```
![Independent1](https://user-images.githubusercontent.com/39111185/218974608-8f1c0d96-0be5-48e2-ade8-9ea6c32c3c16.png)
----------------------------------------------------------------------
![Independent2](https://user-images.githubusercontent.com/39111185/218975123-9e9b2c28-67f5-4631-9933-93e8bcd4109d.png)
----------------------------------------------------------------------
![Independent3](https://user-images.githubusercontent.com/39111185/218975130-e14185b2-e413-4805-823c-6147e95fce0b.png)
----------------------------------------------------------------------
![Independent4](https://user-images.githubusercontent.com/39111185/218975148-d42af656-c567-4fe6-942c-e3fa0206a2e2.png)
----------------------------------------------------------------------
![Independent5](https://user-images.githubusercontent.com/39111185/218975159-071e2cc1-332b-43f7-8227-6b68db077a7a.png)
----------------------------------------------------------------------
![Independent6](https://user-images.githubusercontent.com/39111185/218975172-a2901c6d-9f8b-49de-8079-a04a0074499b.png)
----------------------------------------------------------------------
![Independent7](https://user-images.githubusercontent.com/39111185/218975178-4d3a9b43-5cc5-4be7-b430-0ff89ad26529.png)
----------------------------------------------------------------------
![Independent8](https://user-images.githubusercontent.com/39111185/218975214-404ffc7e-a76a-4f42-ba06-9db534d7fa8b.png)
----------------------------------------------------------------------
![Dependent](https://user-images.githubusercontent.com/39111185/218975770-83c72aeb-a444-43f5-8354-2a1646ba723b.png)
----------------------------------------------------------------------
#### Latitude & Longitude
![KakaoTalk_20230213_093010190](https://user-images.githubusercontent.com/39111185/218976257-dc48f63a-041d-40e5-9725-a5fad6f574ca.png)
----------------------------------------------------------------------
#### Heatmap
![heatmap](https://user-images.githubusercontent.com/39111185/218976287-9f58e620-4554-4568-8e7a-71954b603750.png)
----------------------------------------------------------------------
#### correlation
![correlation값](https://user-images.githubusercontent.com/39111185/218976302-86654479-b493-4cf4-a11d-ebc54d96d82b.png)
----------------------------------------------------------------------

## Feature Engineering
### 파생변수 추가
### 위도&경도 파생변수 생성

## Modeling
### Blending(CatBoost & Lgbm)
### Autogluon
