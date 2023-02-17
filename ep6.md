# episode6

## library install
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

## data load
```
train = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/train_episode6.csv', index_col='id')
test = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/test_episode6.csv', index_col='id')
submission = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/sample_submission_episode6.csv')
original = pd.read_csv('/content/drive/MyDrive/cloud_share/episode6_data/ParisHousing.csv')

```

## EDA
```
train.hist(figsize=(12, 12));
test.hist(figsize=(12, 12));
original.hist(figsize=(12, 12));
```

![train_hist](https://user-images.githubusercontent.com/39111185/219524187-dbda3eae-e4f0-45ff-9024-11894fb36588.png)
![test_hist](https://user-images.githubusercontent.com/39111185/219524191-e45954c4-8598-4385-9b4b-32d03b6bf0d2.png)
![original_hist](https://user-images.githubusercontent.com/39111185/219524196-bf75ae2e-7a79-4fbb-ab04-f17c80788b70.png)



### data concat
```
train = pd.concat([train , original], sort=False)
df = pd.concat([train, test], ignore_index=True)
```

### 이상치 제거
```
df.squareMeters.sort_values(ascending=False)
dict = {6071330 :99999 ,146181:99999 }
df = df.replace({"squareMeters": dict})
```

```
df.floors.sort_values(ascending=False)
dict = {6000 : 100}
df=df.replace({"floors": dict})
```

```
df.made.sort_values(ascending=False)[:10]
dict = { 10000 : 2021 }
df=df.replace({"made": dict})
```

```
df.basement.sort_values(ascending=False)[:10]
dict = { 91992 : 10000 , 91978 : 10000 ,
        89016 : 10000, 26132 : 10000,
        84333 : 10000  , 81851 :10000 }
df=df.replace({"basement": dict})
```

```
df.attic.sort_values(ascending=False)[:20]
dict = { 96381 : 10000, 
         71965 : 10000 , 
         71024 : 10000, 
         71001 : 10000 ,
        38535 : 10000,
         30000 : 10000,
        13779 : 10000
       }
df=df.replace({"attic": dict})
```

```
df.garage.sort_values(ascending=False)[:10]
dict = { 2048 : 1000, 
         9017 : 1000
       }
df=df.replace({"garage": dict})
```

##전처리후 EDA

![이상치제거이후](https://user-images.githubusercontent.com/39111185/219528431-9506ea09-1b33-4e8e-8a01-918901c5ac8a.png)
