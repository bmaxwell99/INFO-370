In [5]: import pandas as pd

In [6]: titanic = pd.read_csv("data/titanic.csv.bz2")

In [7]: titanic.sample(10)
Out[7]: 
      pclass  survived                              name     sex   age  sibsp  parch      ticket     fare cabin embarked boat   body                              home.dest
25         1         0               Birnbaum, Mr. Jakob    male  25.0      0      0       13905  26.0000   NaN        C  NaN  148.0                      San Francisco, CA
482        2         1             Lehmann, Miss. Bertha  female  17.0      0      0     SC 1748  12.0000   NaN        C   12    NaN  Berne, Switzerland / Central City, IA
925        3         0                  Kelly, Mr. James    male  44.0      0      0      363592   8.0500   NaN        S  NaN    NaN                                    NaN
418        2         0              Gilbert, Mr. William    male  47.0      0      0  C.A. 30769  10.5000   NaN        S  NaN    NaN                               Cornwall
1182       3         1  Salkjelsvik, Miss. Anna Kristine  female  21.0      0      0      343120   7.6500   NaN        S    C    NaN                                    NaN
1210       3         0                Skoog, Mr. Wilhelm    male  40.0      1      4      347088  27.9000   NaN        S  NaN    NaN                                    NaN
856        3         1        Healy, Miss. Hanora "Nora"  female   NaN      0      0      370375   7.7500   NaN        Q   16    NaN                                    NaN
961        3         0                Lennon, Miss. Mary  female   NaN      1      0      370371  15.5000   NaN        Q  NaN    NaN                                    NaN
268        1         0          Smith, Mr. Lucien Philip    male  24.0      1      0       13695  60.0000   C31        S  NaN    NaN                         Huntington, WV
895        3         1      Johnson, Miss. Eleanor Ileen  female   1.0      1      1      347742  11.1333   NaN        S   15    NaN                                    NaN

In [8]: titanic.shape
Out[8]: (1309, 14)

In [9]: titanic.columns
Out[9]: 
Index(['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'],
      dtype='object')

In [10]: titanic.sample(10)
Out[10]: 
     pclass  survived                                               name     sex   age  sibsp  parch         ticket      fare cabin embarked boat   body                        home.dest
402       2         1                     Duran y More, Miss. Florentina  female  30.0      1      0  SC/PARIS 2148   13.8583   NaN        C   12    NaN  Barcelona, Spain / Havana, Cuba
296       1         1  Thayer, Mrs. John Borland (Marian Longstreth M...  female  39.0      1      1          17421  110.8833   C68        C    4    NaN                    Haverford, PA
154       1         0                         Hays, Mr. Charles Melville    male  55.0      1      1          12749   93.5000   B69        S  NaN  307.0                     Montreal, PQ
476       2         0                             Lahtinen, Rev. William    male  30.0      1      1         250651   26.0000   NaN        S  NaN    NaN                  Minneapolis, MN
874       3         1                                 Hyman, Mr. Abraham    male   NaN      0      0           3470    7.8875   NaN        S    C    NaN                              NaN
383       2         0                       Cotterill, Mr. Henry "Harry"    male  21.0      0      0          29107   11.5000   NaN        S  NaN    NaN   Penzance, Cornwall / Akron, OH
712       3         0                             Celotti, Mr. Francesco    male  24.0      0      0         343275    8.0500   NaN        S  NaN    NaN                           London
208       1         1    Minahan, Mrs. William Edward (Lillian E Thorpe)  female  37.0      1      0          19928   90.0000   C78        Q   14    NaN                  Fond du Lac, WI
68        1         1                           Chevre, Mr. Paul Romaine    male  45.0      0      0       PC 17594   29.7000    A9        C    7    NaN                    Paris, France
338       2         0                         Beauchamp, Mr. Henry James    male  28.0      0      0         244358   26.0000   NaN        S  NaN    NaN                          England

In [11]: titanic.drop('name', axis=1).sample(10)
Out[11]: 
      pclass  survived     sex   age  sibsp  parch      ticket     fare cabin embarked boat   body            home.dest
332        2         0    male  23.0      0      0  C.A. 31030  10.5000   NaN        S  NaN    NaN             Guernsey
942        3         0    male   NaN      0      0        2624   7.2250   NaN        C  NaN    NaN                  NaN
512        2         0    male  32.5      1      0      237736  30.0708   NaN        C  NaN   43.0         New York, NY
517        2         0    male  26.0      0      0      244368  13.0000    F2        S  NaN    NaN           Boston, MA
1083       3         0    male  28.0      0      0      C 4001  22.5250   NaN        S  NaN  173.0                  NaN
27         1         1  female  19.0      1      0       11967  91.0792   B49        C    7    NaN         Dowagiac, MI
899        3         1  female  27.0      0      2      347742  11.1333   NaN        S   15    NaN                  NaN
161        1         1  female  51.0      1      0       13502  77.9583   D11        S   10    NaN           Hudson, NY
254        1         1    male   NaN      0      0       19988  30.5000  C106        S    3    NaN  Manchester, England
409        2         0    male  36.0      0      0      229236  13.0000   NaN        S  NaN  236.0        Rochester, NY

In [12]: from sklean.model_selection import train_test_split
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-12-e254ffd72cf7> in <module>
----> 1 from sklean.model_selection import train_test_split

ModuleNotFoundError: No module named 'sklean'

In [13]: from sklearn.model_selection import train_test_split

In [14]: train, test = train_test_split(titanic, test_size=0.2)

In [15]: import statsmodels.formula.api as smf

In [16]: m = smf.logit(formula = 'survived ~ 1', data=train)

In [17]: m = smf.logit(formula = 'survived ~ 1', data=train).fit()
Optimization terminated successfully.
         Current function value: 0.665513
         Iterations 4

In [18]: m.summary()
Out[18]: 
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               survived   No. Observations:                 1047
Model:                          Logit   Df Residuals:                     1046
Method:                           MLE   Df Model:                            0
Date:                Thu, 05 Dec 2019   Pseudo R-squ.:               1.161e-10
Time:                        11:05:44   Log-Likelihood:                -696.79
converged:                       True   LL-Null:                       -696.79
Covariance Type:            nonrobust   LLR p-value:                       nan
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.4768      0.064     -7.500      0.000      -0.601      -0.352
==============================================================================
"""

In [19]: yhat = m.predict(train)

In [20]: yhat[:20]
Out[20]: 
1005    0.382999
754     0.382999
1134    0.382999
849     0.382999
726     0.382999
556     0.382999
66      0.382999
94      0.382999
11      0.382999
1026    0.382999
870     0.382999
8       0.382999
36      0.382999
641     0.382999
48      0.382999
504     0.382999
858     0.382999
751     0.382999
615     0.382999
1234    0.382999
dtype: float64

In [21]: Out[20]: 
1005    0.382999
754     0.382999
1134    0.382999
849     0.382999
726     0.382999
556     0.382999
66      0.382999
94      0.382999
11      0.382999
1026    0.382999
870     0.382999
8       0.382999
36      0.382999
641     0.382999
48      0.382999
504     0.382999
858     0.382999
751     0.382999
615     0.382999
1234    0.382999
dtype: float64

In [21]: 
    ...:   File "<ipython-input-21-12a648f36551>", line 1
    Out[20]:
             ^
SyntaxError: invalid syntax


In [22]:   File "<ipython-input-22-6431eaa66072>", line 1
    754     0.382999
                   ^
SyntaxError: invalid syntax


In [23]:   File "<ipython-input-23-268bb9d81d08>", line 1
    1134    0.382999
                   ^
SyntaxError: invalid syntax


In [24]:   File "<ipython-input-24-44d0f1980625>", line 1
    849     0.382999
                   ^
SyntaxError: invalid syntax


In [25]:   File "<ipython-input-25-356021bde3bd>", line 1
    726     0.382999
                   ^
SyntaxError: invalid syntax


In [26]:   File "<ipython-input-26-2d6b92ab92b8>", line 1
    556     0.382999
                   ^
SyntaxError: invalid syntax


In [27]:   File "<ipython-input-27-1b7b26ebdd05>", line 1
    66      0.382999
                   ^
SyntaxError: invalid syntax


In [28]:   File "<ipython-input-28-37feaea0d51b>", line 1
    94      0.382999
                   ^
SyntaxError: invalid syntax


In [29]:   File "<ipython-input-29-84bed333c3d2>", line 1
    11      0.382999
                   ^
SyntaxError: invalid syntax


In [30]:   File "<ipython-input-30-6665d23d46ff>", line 1
    1026    0.382999
                   ^
SyntaxError: invalid syntax


In [31]:   File "<ipython-input-31-e56009ae1803>", line 1
    870     0.382999
                   ^
SyntaxError: invalid syntax


In [32]:   File "<ipython-input-32-5641005b8d04>", line 1
    8       0.382999
                   ^
SyntaxError: invalid syntax


In [33]:   File "<ipython-input-33-fe5413b67c19>", line 1
    36      0.382999
                   ^
SyntaxError: invalid syntax


In [34]:   File "<ipython-input-34-9985c238b5cb>", line 1
    641     0.382999
                   ^
SyntaxError: invalid syntax


In [35]:   File "<ipython-input-35-947e7bea861e>", line 1
    48      0.382999
                   ^
SyntaxError: invalid syntax


In [36]:   File "<ipython-input-36-8c94c7b89f32>", line 1
    504     0.382999
                   ^
SyntaxError: invalid syntax


In [37]:   File "<ipython-input-37-af3b531e8505>", line 1
    858     0.382999
                   ^
SyntaxError: invalid syntax


In [38]:   File "<ipython-input-38-45b8010960ed>", line 1
    751     0.382999
                   ^
SyntaxError: invalid syntax


In [39]:   File "<ipython-input-39-2b11e1ac3208>", line 1
    615     0.382999
                   ^
SyntaxError: invalid syntax


In [40]:   File "<ipython-input-40-9e272532e92a>", line 1
    1234    0.382999
                   ^
SyntaxError: invalid syntax


In [41]: ---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-41-676ff4f042a6> in <module>
----> 1 dtype: float64

NameError: name 'float64' is not defined

In [42]: 
In [42]: 
In [43]: train.survived.mean()
Out[43]: 0.38299904489016234

In [44]: np.random.seed(1)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-44-d48aee91a53d> in <module>
----> 1 np.random.seed(1)

NameError: name 'np' is not defined

In [45]: import numpy as np

In [46]: np.random.seed(1)

In [47]: train, test = train_test_split(titanic, test_size=0.2)

In [48]: m = smf.logit(formula = 'survived ~ 1', data=train).fit()
Optimization terminated successfully.
         Current function value: 0.662230
         Iterations 4

In [49]: yhat = m.predict(train)

In [50]: yhat[1]
Out[50]: 0.37631327602674314

In [51]: yhat = m.redict(test)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-51-a1d6a14bfafb> in <module>
----> 1 yhat = m.redict(test)

~/anaconda3/lib/python3.7/site-packages/statsmodels/base/wrapper.py in __getattribute__(self, attr)
     33             pass
     34 
---> 35         obj = getattr(results, attr)
     36         data = results.model.data
     37         how = self._wrap_attrs.get(attr)

AttributeError: 'LogitResults' object has no attribute 'redict'

In [52]: yhat = m.predict(test)

In [53]: yhat[1]
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-53-dbe1af4242e5> in <module>
----> 1 yhat[1]

~/anaconda3/lib/python3.7/site-packages/pandas/core/series.py in __getitem__(self, key)
   1069         key = com.apply_if_callable(key, self)
   1070         try:
-> 1071             result = self.index.get_value(self, key)
   1072 
   1073             if not is_scalar(result):

~/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_value(self, series, key)
   4728         k = self._convert_scalar_indexer(k, kind="getitem")
   4729         try:
-> 4730             return self._engine.get_value(s, k, tz=getattr(series.dtype, "tz", None))
   4731         except KeyError as e1:
   4732             if len(self) > 0 and (self.holds_integer() or self.is_boolean()):

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_value()

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_value()

pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 1

In [54]: yhat.shape
Out[54]: (262,)

In [55]: type(yhat)
Out[55]: pandas.core.series.Series

In [56]: yhat.iloc[0]
Out[56]: 0.37631327602674314

In [57]: yhat = m.predict(test)

In [58]: yhat.iloc[0]
Out[58]: 0.37631327602674314

In [59]: yhat.iloc[:10]
Out[59]: 
201     0.376313
115     0.376313
255     0.376313
1103    0.376313
195     0.376313
1281    0.376313
1138    0.376313
288     0.376313
270     0.376313
248     0.376313
dtype: float64

In [60]: yhat = m.predict(test) > 0.5

In [61]: yhat.iloc[:5]
Out[61]: 
201     False
115     False
255     False
1103    False
195     False
dtype: bool

In [62]: pd.crosstab(test.survived, yhat)
Out[62]: 
col_0     False
survived       
0           156
1           106

In [63]: 156/(156 + 106)
Out[63]: 0.5954198473282443

In [64]: np.mean(yhat == y)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-64-a5124056405d> in <module>
----> 1 np.mean(yhat == y)

NameError: name 'y' is not defined

In [65]: np.mean(yhat == titanic.survived)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-65-466690a24f3a> in <module>
----> 1 np.mean(yhat == titanic.survived)

~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/__init__.py in wrapper(self, other, axis)
   1140 
   1141         elif isinstance(other, ABCSeries) and not self._indexed_same(other):
-> 1142             raise ValueError("Can only compare identically-labeled " "Series objects")
   1143 
   1144         elif is_categorical_dtype(self):

ValueError: Can only compare identically-labeled Series objects

In [66]: np.mean(yhat == titanic.survived.astype('int'))
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-66-505b4f1d34da> in <module>
----> 1 np.mean(yhat == titanic.survived.astype('int'))

~/anaconda3/lib/python3.7/site-packages/pandas/core/ops/__init__.py in wrapper(self, other, axis)
   1140 
   1141         elif isinstance(other, ABCSeries) and not self._indexed_same(other):
-> 1142             raise ValueError("Can only compare identically-labeled " "Series objects")
   1143 
   1144         elif is_categorical_dtype(self):

ValueError: Can only compare identically-labeled Series objects

In [67]: np.mean(yhat.values == titanic.survived.values)
/home/otoomet/anaconda3/bin/ipython3:1: DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
  #!/home/otoomet/anaconda3/bin/python
Out[67]: 0.0

In [68]: titanic.columns
Out[68]: 
Index(['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'],
      dtype='object')

In [69]: train.sex.value_counts()
Out[69]: 
male      676
female    371
Name: sex, dtype: int64

In [70]: m = smf.logit(formula = 'survived ~ sex', data=train).fit()
Optimization terminated successfully.
         Current function value: 0.531167
         Iterations 5

In [71]: m.summary()
Out[71]: 
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               survived   No. Observations:                 1047
Model:                          Logit   Df Residuals:                     1045
Method:                           MLE   Df Model:                            1
Date:                Thu, 05 Dec 2019   Pseudo R-squ.:                  0.1979
Time:                        11:42:03   Log-Likelihood:                -556.13
converged:                       True   LL-Null:                       -693.36
Covariance Type:            nonrobust   LLR p-value:                 1.218e-61
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept       0.8900      0.114      7.788      0.000       0.666       1.114
sex[T.male]    -2.3156      0.150    -15.427      0.000      -2.610      -2.021
===============================================================================
"""

In [72]: yhat = m.predict(test) > 0.5

In [73]: yhat.iloc[:10]
Out[73]: 
201     False
115     False
255      True
1103    False
195      True
1281    False
1138    False
288      True
270      True
248     False
dtype: bool

In [74]: pd.crosstab(test.survived, yhat)
Out[74]: 
col_0     False  True 
survived              
0           137     19
1            30     76

In [75]: (137 + 19)/(137 + 19 + 30 + 76)
Out[75]: 0.5954198473282443

In [76]: (137 + 76)/(137 + 19 + 30 + 76)
Out[76]: 0.8129770992366412

In [77]: titanic.columns
Out[77]: 
Index(['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'],
      dtype='object')

In [78]: 

In [78]: train.age[:11]
Out[78]: 
62      46.0
503     19.0
745     30.0
1154     NaN
826      1.0
1211    45.0
154     55.0
471     45.0
596     31.0
641      3.0
188     51.0
Name: age, dtype: float64

In [79]: 

In [79]: m = smf.logit(formula = 'survived ~ sex + age', data=train).fit()
Optimization terminated successfully.
         Current function value: 0.532826
         Iterations 5

In [80]: m.summary()
Out[80]: 
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               survived   No. Observations:                  826
Model:                          Logit   Df Residuals:                      823
Method:                           MLE   Df Model:                            2
Date:                Thu, 05 Dec 2019   Pseudo R-squ.:                  0.2086
Time:                        11:50:43   Log-Likelihood:                -440.11
converged:                       True   LL-Null:                       -556.15
Covariance Type:            nonrobust   LLR p-value:                 4.038e-51
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept       1.2564      0.218      5.765      0.000       0.829       1.683
sex[T.male]    -2.3691      0.169    -13.994      0.000      -2.701      -2.037
age            -0.0081      0.006     -1.385      0.166      -0.019       0.003
===============================================================================
"""

In [81]: yhat = m.predict(test) > 0.5

In [82]: pd.crosstab(test.survived, yhat)
Out[84]: 
col_0     False  True 
survived              
0           141     15
1            38     68

In [85]: (141 + 68)/(141 + 15 + 38 + 68)
Out[85]: 0.7977099236641222

In [86]: train['child'] = train.age < 12
/home/otoomet/anaconda3/bin/ipython3:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  #!/home/otoomet/anaconda3/bin/python

In [87]: train = train.copy()

In [88]: test = test.copy()

In [89]: train['child'] = train.age < 12

In [90]: test['child'] = test.age < 12

In [91]: m = smf.logit(formula = 'survived ~ sex + child', data=train).fit()
Optimization terminated successfully.
         Current function value: 0.524754
         Iterations 5

In [92]: m.summary()
Out[92]: 
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               survived   No. Observations:                 1047
Model:                          Logit   Df Residuals:                     1044
Method:                           MLE   Df Model:                            2
Date:                Thu, 05 Dec 2019   Pseudo R-squ.:                  0.2076
Time:                        11:59:06   Log-Likelihood:                -549.42
converged:                       True   LL-Null:                       -693.36
Covariance Type:            nonrobust   LLR p-value:                 3.080e-63
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept         0.8172      0.116      7.053      0.000       0.590       1.044
sex[T.male]      -2.3222      0.152    -15.326      0.000      -2.619      -2.025
child[T.True]     1.0418      0.285      3.649      0.000       0.482       1.601
=================================================================================
"""

In [93]: yhat = m.predict(test) > 0.5

In [94]: pd.crosstab(test.survived, yhat)
Out[94]: 
col_0     False  True 
survived              
0           137     19
1            30     76

In [95]: (137 + 76)/(137 + 76 + 19 + 30)
Out[95]: 0.8129770992366412

In [96]: train.columns
Out[96]: 
Index(['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
       'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest', 'child'],
      dtype='object')

In [97]: train.head()
Out[97]: 
      pclass  survived                                    name     sex   age  sibsp  parch           ticket    fare cabin embarked boat  body                             home.dest  child
62         1         0             Chaffee, Mr. Herbert Fuller    male  46.0      1      0      W.E.P. 5734  61.175   E31        S  NaN   NaN                            Amenia, ND  False
503        2         1               Mellors, Mr. William John    male  19.0      0      0        SW/PP 751  10.500   NaN        S    B   NaN                       Chelsea, London  False
745        3         1  Daly, Miss. Margaret Marcella "Maggie"  female  30.0      0      0           382650   6.950   NaN        Q   15   NaN      Co Athlone, Ireland New York, NY  False
1154       3         0                Rogers, Mr. William John    male   NaN      0      0  S.C./A.4. 23567   8.050   NaN        S  NaN   NaN                                   NaN  False
826        3         0         Goodwin, Master. Sidney Leonard    male   1.0      5      2          CA 2144  46.900   NaN        S  NaN   NaN  Wiltshire, England Niagara Falls, NY   True

In [98]: train.sample(20)
Out[98]: 
      pclass  survived                                               name     sex   age  sibsp  parch  ...     fare  cabin embarked   boat   body                                       home.dest  child
465        2         1       Jerwan, Mrs. Amin S (Marie Marthe Thuillard)  female  23.0      0      0  ...  13.7917      D        C     11    NaN                                    New York, NY  False
1098       3         0                         Palsson, Miss. Stina Viola  female   3.0      3      1  ...  21.0750    NaN        S    NaN    NaN                                             NaN   True
486        2         0                  Leyson, Mr. Robert William Norman    male  24.0      0      0  ...  10.5000    NaN        S    NaN  108.0                                             NaN  False
797        3         0                                 Farrell, Mr. James    male  40.5      0      0  ...   7.7500    NaN        Q    NaN   68.0  Aughnacliff, Co Longford, Ireland New York, NY  False
1036       3         1   Moubarek, Mrs. George (Omine "Amenia" Alexander)  female   NaN      0      2  ...  15.2458    NaN        C      C    NaN                                             NaN  False
935        3         1                           Kink-Heilmann, Mr. Anton    male  29.0      3      1  ...  22.0250    NaN        S      2    NaN                                             NaN  False
662        3         0                                  Badt, Mr. Mohamed    male  40.0      0      0  ...   7.2250    NaN        C    NaN    NaN                                             NaN  False
501        2         1                  Mellinger, Miss. Madeleine Violet  female  13.0      0      1  ...  19.5000    NaN        S     14    NaN                        England / Bennington, VT  False
613        3         1                        Albimona, Mr. Nassef Cassem    male  26.0      0      0  ...  18.7875    NaN        C     15    NaN                        Syria Fredericksburg, VA  False
257        1         1                    Schabert, Mrs. Paul (Emma Mock)  female  35.0      1      0  ...  57.7500    C28        C     11    NaN                                    New York, NY  False
932        3         0                                  Kink, Miss. Maria  female  22.0      2      0  ...   8.6625    NaN        S    NaN    NaN                                             NaN  False
382        2         0  Corey, Mrs. Percy C (Mary Phyllis Elizabeth Mi...  female   NaN      0      0  ...  21.0000    NaN        S    NaN    NaN               Upper Burma, India Pittsburgh, PA  False
570        2         1                                Toomey, Miss. Ellen  female  50.0      0      0  ...  10.5000    NaN        S      9    NaN                                Indianapolis, IN  False
798        3         1                                  Finoli, Mr. Luigi    male   NaN      0      0  ...   7.0500    NaN        S     15    NaN                          Italy Philadelphia, PA  False
980        3         1                           Lundin, Miss. Olga Elida  female  23.0      0      0  ...   7.8542    NaN        S     10    NaN                                             NaN  False
680        3         0                                  Boulos, Mr. Hanna    male   NaN      0      0  ...   7.2250    NaN        C    NaN    NaN                                           Syria  False
1277       3         1                               Vartanian, Mr. David    male  22.0      0      0  ...   7.2250    NaN        C  13 15    NaN                                             NaN  False
508        2         0                               Moraweck, Dr. Ernest    male  54.0      0      0  ...  14.0000    NaN        S    NaN    NaN                                   Frankfort, KY  False
92         1         1          Dick, Mrs. Albert Adrian (Vera Gillespie)  female  17.0      1      0  ...  57.0000    B20        S      3    NaN                                     Calgary, AB  False
522        2         0                                 Otter, Mr. Richard    male  39.0      0      0  ...  13.0000    NaN        S    NaN    NaN                          Middleburg Heights, OH  False

[20 rows x 15 columns]
