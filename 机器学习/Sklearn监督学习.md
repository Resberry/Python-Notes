### 理论知识
#### 学习目标
利用一组带有标签的数据，学习从输入到输出的映射，然后将这种映射关系应用到未知数据上，达到分类或回归的目的。

分类：当输出是离散的，学习任务为分类任务。

回归：当输出是连续的，学习任务为回归任务。

* 训练集(training set):顾名思义用来训练模型的已标注数据，用来建立模型，发现规律。
* 测试集(testing set):也是已标注数据，通常做法是将标注隐藏，输送给训练好的模型，通过结果与真实标注进行对比，评估模型的学习能力。

#### 评价标准
* 精确率：精确率是针对我们预测结果而言的，（以二分类为例）它表示的是预测为正的样本中有多少是真正的正样本。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是TP/(TP+FP)。
* 召回率：是针对我们原来的样本而言的，它表示的是样本中的正例有多少被预测正确了。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)，也就是TP/(TP+FN)。

Sklearn库中的分类算法并未被统一封装在一个子模块中，因此对分类算法的import方式各有不同。

#### Sklearn提供的分类函数
* k近邻（knn）
* 朴素贝叶斯（naivebayes）
* 支持向量机（svm）
* 决策树 （decision tree）
* 神经网络模型（Neural networks）等

这其中有线性分类器，也有非线性分类器。

#### Sklearn提供的回归函数
主要被封装在两个子模块中，分别是`sklearn.linear_model`和`sklearn.preprocessing`。

`sklearn.linear_modlel`封装的是一些线性函数，线性回归函数包括有：
* 普通线性回归函数（ LinearRegression ）
* 岭回归（Ridge）
* Lasso（Lasso）

非线性回归函数：如多项式回归（PolynomialFeatures）则通过`sklearn.preprocessing`子模块进行调用。

### 分类
#### K近邻分类器（KNN）
通过计算待分类数据点，与已有数据集中的所有数据点的距离。取距离最小的前K个点，根据“少数服从多数“的原则，将这个数据点划分为出现次数最多的那个类别。
```python
sklearn.neighbors.KNeighborsClassifier
```

##### 主要参数
* n_neighbors：用于指定分类器中K的大小(默认值为5，注意与kmeans的区别)
* weights：设置选中的K个点对分类结果影响的权重（默认值为平均权重“uniform”，可以选择“distance”代表越近的点权重越高，或者传入自己编写的以距离为参数的权重计算函数）
* algorithm：设置用于计算临近点的方法，因为当数据量很大的情况下计算当前点和所有点的距离再选出最近的k各点，这个计算量是很费时的，所以（选项中有ball_tree、kd_tree和brute，分别代表不同的寻找邻居的优化算法，默认值为auto，根据训练数据自动选择）

##### 程序实例
```python
from sklearn.neighbors import KNeighborsClassifier
X = [[0], [1], [2], [3]]      #创建一组数据
y = [0, 0, 1, 1]              #数据的标签
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y) 
print(neigh.predict([[1.1]])) #对未知样本进行分类
输出：[0]
```

#### 决策树
决策树是一种树形结构的分类器，通过顺序询问分类点的属性决定分类点最终的类别。通常根据特征的信息增益或其他指标，构建一棵决策树。在分类时，只需要按照决策树中的结点依次进行判断，即可得到样本所属类别。
```python
sklearn.tree.DecisionTreeClassifier
```

##### 主要参数
* criterion ：用于选择属性的准则，可以传入“gini”代表基尼系数，或者“entropy”代表信息增益。
* max_features ：表示在决策树结点进行分裂时，从多少个特征中选择最优特征。可以设定固定数目、百分比或其他标准。它的默认值是使用所有特征个数。

##### 程序实例
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score        #入计算交叉验证值的函数

clf = DecisionTreeClassifier()
iris = load_iris()
print(cross_val_score(clf, iris.data, iris.target, cv=10)) #使用10折交叉验证，得到最终的交叉验证得分
输出：array([ 1. , 0.93..., 0.86..., 0.93..., 0.93...,0.93..., 0.93..., 1. , 0.93..., 1. ])

clf.fit(X, y)
clf.predict(x)
```

##### 交叉验证
将数据集分为k折。将每一折都当做一次测试集，其余k-1折当做训练集，这样循环k次，最后将k次结果求平均值。

优点：
* 交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表现，可以在一定程度上减小过拟合。
* 可以从有限的数据中获取尽可能多的有效信息。

#### 朴素贝叶斯
朴素贝叶斯分类器是一个以贝叶斯定理为基础的多分类的分类器，一般在小规模数据上的表现很好，适合进行多分类任务。

对于给定数据，首先基于特征的条件独立性假设，学习输入输出的联合概率分布，然后基于此模型，对给定的输入x，利用贝叶斯定理求出后验概率最大的输出y。
```python
sklearn.naive_bayes.GussianNB #高斯朴素贝叶斯分类器
sklearn.naive_bayes.MultinomialNB #针对多项式模型的朴素贝叶斯分类器
sklearn.naive_bayes.BernoulliNB #针对多元伯努利模型的朴素贝叶斯分类器
```

以高斯朴素贝叶斯分类器为例，其参数有：
* priors ：给定各个类别的先验概率。如果为空，则按训练数据的实际情况进行统计；如果给定先验概率，则在训练过程中不能更改。

##### 程序实例
```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) 
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB(priors=None)
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
输出：[1]
```

##### 实例讲解
* 可穿戴式设备的流行，让我们可以更便利地使用传感器获取人体的各项数据，甚至生理数据。
* 当传感器采集到大量数据后，我们就可以通过对数据进行分析和建模，通过各项特征的数值进行用户状态的判断，根据用户所处的状态提供给用户更加精准、便利的服务。
* 我们现在收集了来自 A,B,C,D,E 5位用户的可穿戴设备上的传感器数据，每位用户的数据集包含一个特征文件（a.feature）和一个标签文件（a.label）。
* 特征文件中每一行对应一个时刻的所有传感器数值，标签文件中每行记录了和特征文件中对应时刻的标记过的用户姿态，两个文件的行数相同，相同行之间互相对应。

##### 程序编写
*#需pip install pandas*
```python
import pandas as pd
import numpy as np  
 
from sklearn.preprocessing import Imputer             #预处理模块
from sklearn.cross_validation import train_test_split #自动生成训练集和测试集的模块
from sklearn.metrics import classification_report     #预测结果评估模块
   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
 
def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))                               #创建一个n维数组变量，目前有0行41列
    label = np.ndarray(shape=(0,1))
    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        #读取特征文件的内容。其中指定分隔符为逗号、缺失值为问号且文件不包含表头行。
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #使用平均值对缺失数据进行补全
        imp.fit(df)                                                  #用设定好的Imputer训练读取器
        df = imp.transform(df)                                       #生成预处理的结果
        feature = np.concatenate((feature, df))                      #将预处理后的数据加入feature矩阵；依次遍历完所有特征文件
     
    for file in label_paths:
        df = pd.read_table(file, header=None)                        #读取标签数据
        label = np.concatenate((label, df))                          #直接将数据加入label矩阵
         
    label = np.ravel(label)                                          #将label矩阵化为一维向量
    return feature, label
 
if __name__ == '__main__':
    ''' 设置数据路径 '''
    featurePaths = ['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    labelPaths = ['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    ''' 读入数据 '''
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4]) #前4个数据作为训练集读入
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])   #最后一个数据作为测试集读入
     
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')
     
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
     
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
     
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))
```

### 回归
#### 线性回归
```python
sklearn.linear_model.LinearRegression
```
##### 实例讲解
根据已知的房屋成交价和房屋的尺寸进行线性回归，继而可以对已知房屋尺寸，而未知房屋成交价格的实例进行成交价格的预测。
##### 程序设计
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

datasets_X = []
datasets_Y = []
f = open('prices.txt','r')
lines = f.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
 
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
 
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
 
linear = linear_model.LinearRegression()
linear.fit(datasets_X, datasets_Y)

plt.scatter(datasets_X, datasets_Y, color = 'red')
plt.plot(X, linear.predict(X), color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
```

#### 非线性回归（多项式回归）
这里的多项式回归实际上是先将变量X处理成多项式特征，然后使用线性模型学习多项式特征的参数，以达到多项式回归的目的。

例如：X = [x_1, x_2]
1. 使用PolynomialFeatures构造X的二次多项式特征X_Poly：
X_Poly = [x_1, x_2, x_1x_2, x_1^2, x_2^2]
2. 使用linear_model学习X_Poly和y之间的映射关系，即：
w_1x_1+w_2x_2+w_3x_1x_2+w_4x_1^2+w_5x_2^2=y
```python
sklearn.preprocessing.PolynomialFeatures
```

##### 实例讲解
##### 程序设计
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

datasets_X = []
datasets_Y = []
fr = open('prices.txt','r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
 
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
 
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
  
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)
 
plt.scatter(datasets_X, datasets_Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
```

#### 岭回归
当数据中出现多重共线性时，传统线性回归运用的最小二乘法将失去稳定性，所以需要岭回归。

系数矩阵求解方法：
* 最小二乘法：w=[X^T*X]^(-1)*X^T*y
* 岭回归：w=[X^T*X+αI]^(-1)*X^T*y

岭回归(ridge regression)是一种专用于共线性数据分析的有偏估计回归方法。是一种改良的最小二乘估计法，对某些数据的拟合要强于最小二乘法。
```python
sklearn.linear_model.Ridge
```
##### 主要参数
* alpha：正则化因子，对应于损失函数中的α
* fit_intercept：表示是否计算截距
* solver：设置计算参数的方法，可选参数'auto'、'svd'、'sag'等

##### 实例讲解
数据为某路口的交通流量监测数据，记录全年小时级别的车流量。根据已有的数据创建多项式特征，使用岭回归模型代替一般的线性模型，对车流量的信息进行多项式回归。

##### 程序设计
```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data=np.genfromtxt('data.txt')   #直接读取文件中数据化为矩阵
X=data[:,:4]                     #属性
y=data[:,4]                      #车流量
poly=PolynomialFeatures(6)
X=poly.fit_transform(X)
train_set_X, test_set_X , train_set_y, test_set_y = 
    cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)
#将所有数据划分为训练集和测试集，test_size表示测试集的比例
clf=Ridge(alpha=1.0,fit_intercept = True) 
clf.fit(train_set_X,train_set_y)
clf.score(test_set_X,test_set_Y) #计算回归曲线的拟合优度

start=200 
end=300 
y_pre=clf.predict(X)             #是调用predict函数的拟合值
time=np.arange(start,end)
plt.plot(time,y[start:end],'b', label="real")
plt.plot(time,y_pre[start:end],'r', label='predict')
plt.legend(loc='upper left')     #设置图例的位置
plt.show()   
```
