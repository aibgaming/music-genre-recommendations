# Music Genre Recommendations
Uses a Decision Tree Classifier to make music genre recommendations and visualize the tree created

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
```


```python
music_data = pd.read_csv("music.csv")
music_data
```






```python
music_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18 entries, 0 to 17
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   age     18 non-null     int64 
     1   gender  18 non-null     int64 
     2   genre   18 non-null     object
    dtypes: int64(2), object(1)
    memory usage: 560.0+ bytes
    


```python
music_data.hist(bins = 50, figsize=(10,5))
```




    array([[<AxesSubplot:title={'center':'age'}>,
            <AxesSubplot:title={'center':'gender'}>]], dtype=object)




    
![png](output_3_1.png)
    



```python
X = music_data.drop(columns = ["genre"])
X.head()
```






```python
y = music_data["genre"]
y.head()
```




    0    HipHop
    1    HipHop
    2    HipHop
    3      Jazz
    4      Jazz
    Name: genre, dtype: object




```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2)   # 20% of the dataset will be for testing
```


```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```







```python
predictions = model.predict(X_test)
predictions
```




    array(['Dance', 'Classical', 'Jazz', 'Dance'], dtype=object)




```python
score = accuracy_score(y_test, predictions)
score #Varies from 25% to 100% depends on how much data is used to train model
```




    1.0




```python
joblib.dump(model, "music-recommender.joblib")
```




    ['music-recommender.joblib']




```python
loaded_model = joblib.load("music-recommender.joblib")
loaded_model.predict(X_test)
```




    array(['Dance', 'Classical', 'Jazz', 'Dance'], dtype=object)




```python
from sklearn import tree
import matplotlib.pyplot as plt
```

Visualize the decision tree created by the model


```python
tree.export_graphviz(model, out_file="music-recommender-dot",
                    feature_names= ['age','gender'],
                    class_names=sorted(y.unique()),
                    label = "all",
                    rounded =True,
                    filled=True)
```


```python
plt.figure(figsize =(10,5))
_ = tree.plot_tree(model, feature_names= ['age','gender'],
                  filled = True,
                   class_names=sorted(y.unique()),
                  fontsize=6,
                  rounded=True)
plt.show()
```


    
![png](output_15_0.png)
    

