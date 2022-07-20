# Data pre-processing

### Normalization: Min-Max Scaling

- All the values will be between [0, 1]

$$
x^{i}_{norm} = \frac{x^{i} - x_{min}}{x_{max}-x_{min}}
$$

### Normalization: Standardization

- Centers at mean 0 and scaling S.T std = 0
- If the distribution of the feature is not normal, after standardization it won’t be normal
- This normalization technique helps certain optimization techniques such as SGD.

$$
x^{i}_{std} = \frac{x^{i}-\mu_{x}}{\sigma_{x}}
$$

 

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### Working with Categorical Data

- Ordinal data is when order matters; size medium is less than large.
    - use a mapping dict to convert categorical data to ordinal data
- Nominal data is data that does not contain inherent order
    - you can use the scikit-learn LabelEncoder
    - using one-hot encoding is actually better
        - if the label encoder encodes red as 1 and blue as 2. Does this mean that there is a distance difference between the two? of course not. That’s why one-hot almost always outperforms label encoders
        - We don’t want the ML algorithm to infer an order from the label encoder!
        - use the drop_first option as it will help with optimization and reduce the number of features.
    
    ```python
    from sklearn import LabelEncoder
    le = LabelEncoder()
    df['cat'] = le.fit_transform(df['cat'])
    
    #one-hot
    pd.get_dummies(df, drop_first=True)
    
    ```