### 安装Sklearn
pip按顺序安装：numpy、scipy、matplotlib、sklearn。

*#sklearn是package不是module，所以引入时不能`import sklearn`，而是`from sklearn import xxx`*

无监督学习：无法给出训练样本的标签，即不知道它们属于哪个类型。

<br>

### K-Means聚类
#### 基本思想
K-Means算法以k为参数，把n个对象分成k个簇，使簇内具有较高的相似度，而簇间的相似度较低。

其处理过程如下：
1. 随机选择k个点作为初始的聚类中心。
2. 对于剩下的点，根据其与聚类中心的距离，将其归入最近的簇。
3. 对每个簇，计算所有点的均值作为新的聚类中心。
4. 重复2、3直到聚类中心不再发生改变。

#### 实例讲解
数据介绍：现有1999年全国31个省份城镇居民家庭平均每人全年消费性支出的八个主要变量数据，根据数据对省份进行聚类。

数据形如：
```python
北京 2959 730.79 749.41 513.34 467.87 1141.82 478.42 457.64
天津 2460 495.47 697.33 302.87 284.19 735.97 570.84 305.08
河北 1496 515.9 362.37 285.32 272.95 540.58 364.91 188.63
```
#### 程序设计
```python
import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath): #从文件中读取数据
	f=open(filePath,'r')
	lines=f.readlines()
	retData=[]
	retCityName=[]
	for line in lines:
		items=line.strip().split()
		retCityName.append(items[0])
		retData.append([float(items[i]) for i in range(1,len(items))])
	return retData,retCityName
 
data,cityName=loadData('F:/city.txt')
#data的形式为一个大列表，大列表中的元素为31个小列表包裹着的每个城市的八个变量数据
km=KMeans(n_clusters=3)                     #初始化实例，n_clusters为聚类中心的个数
label=km.fit_predict(data)                  #计算簇中心以及为簇分配序号，计算每组数据分别属于哪个簇
expenses=np.sum(km.cluster_centers_,axis=1) #计算每个簇中心的指数（即每个簇的平均消费），axis=1代表列压缩
CityCluster=[[],[],[]]
for i in range(len(cityName)):
	CityCluster[label[i]].append(cityName[i]) #根据簇序号将城市分配到不同簇中
for i in range(len(CityCluster)):
	print("Expenses:%.2f"%expenses[i])
	print(CityCluster[i])
```
#### 运行结果
```python
Expenses:3827.87
['河北','山西','内蒙古','辽宁','吉林','黑龙江','安徽','江西','山东','河南','湖北','贵州','陕西','甘肃','青海','宁夏','新疆']
Expenses:5113.54
['天津','江苏','浙江','福建','湖南','广西','海南','重庆','四川','云南','西藏']
Expenses:7754.66
['北京','上海','广东']
```
*#K-Means计算距离时默认使用欧式距离，且没有有关的参数，如果想更换，可以修改K-Means的源代码：在euclidean_distances函数处，使用scipy.spatial.distance.cdist(A,B,metric=' ')，通过metric参数进行更换。*

<br>

### DBSCAN密度聚类
#### 基本思想
特点：聚类的时候不需要预先指定簇的个数，最终的簇的个数不定。

DBSCAN算法将数据点分为三类：
* 核心点：在半径Eps内含有超过MinPts数目的点
* 边界点：在半径Eps内点的数量小于MinPts，但是落在核心点的邻域内
* 噪音点：既不是核心点也不是边界点的点

DBSCAN算法流程：
1. 将所有点标记为核心点、边界点或噪声点；
2. 删除噪声点；
3. 为距离在Eps之内的所有核心点之间赋予一条边；
4. 每组连通的核心点形成一个簇；
5. 将每个边界点指派到一个与之关联的核心点的簇中（哪一个核心点的半径范围之内）。

DBSCAN主要参数：
* eps：两个样本被看作邻居节点的最大距离
* min_samples：簇的最小样本数
* metric：距离计算方式

`sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean')`

#### 实例讲解
现有大学校园网的日志数据，290条大学生的校园网使用情况数据。利用上网时段，分析学生上网的模式。

#### 程序设计
```python
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

 onlinetimes=[]
#此处省略读取文件的过程，最终onlinetimes为一个列表，列表中每个元素为一个小列表，小列表中存放着单人的上网起始时间和上网时长
real_X=np.array(onlinetimes).reshape((-1,2)) 
#reshape()为改变矩阵形状，参数为新矩阵的行列数，参数-1表示不确定有多少行，由原矩阵和给出的列数自动进行计算
 X=real_X[:,0:1]                                           #取矩阵real_X的第一列，即全部学生的上网起始时间
 
db=skc.DBSCAN(eps=0.01,min_samples=20).fit(X)
labels = db.labels_                                        #获得每个数据的簇序号
 
print('Labels:')
print(labels)
raito=len(labels[labels[:] == -1]) / len(labels)           #计算噪声点所占比例
print('Noise raito:',format(raito, '.2%'))
 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #计算簇的数量
print('Estimated number of clusters: %d' % n_clusters)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
#metrics.silhouette_score()为轮廓系数，取值范围[-1,1]，同类样本距离越近、不同类样本距离越远，分数越高
 
for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels == i].flatten()))                  #打印每个簇的数据；flatten()为numpy中的函数，将一个数组转化为一维数组
     
plt.hist(X,24) #画出柱状图，24为横坐标长度
plt.show()     #展示图形
```

<br>

### 降维：PCA算法
#### 基本思想
主成分分析（Principal Component Analysis，PCA）是最常用的一种降维方法，可以把具有相关性的高维变量合成为线性无关的低维变量，称为主成分，主成分能够尽可能保留原始数据的信息。
sklearn.decomposition.PCA

主要参数有：
* n_components：指定主成分的个数，即降维后数据的维度
* svd_solver：设置特征值分解的方法，默认为'auto'，其他可选有'full', 'arpack', 'randomized'
#### 实例讲解
已知鸢尾花数据是4维的，共三类样本。使用PCA实现对鸢尾花数据进行降维，实现在二维平面上的可视化。

#### 程序设计
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
 
data = load_iris()
Y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)
 
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
 
for i in range(len(reduced_X)):
    if Y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif Y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
 
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
```

<br>

### 降维：NMF算法
#### 基本思想
非负矩阵分解（Non-negative Matrix Factorization，NMF）是在矩阵中所有元素均为非负数约束条件之下的矩阵分解方法。

基本思想：给定一个非负矩阵V，NMF能够找到一个非负矩阵W和一个非负矩阵H，使得矩阵W和H的乘积近似等于矩阵V中的值。

* W矩阵：基础图像矩阵，相当于从原矩阵V中抽取出来的特征
* H矩阵：系数矩阵。
* NMF能够广泛应用于图像分析、文本挖掘和语音处理等领域。
```python
sklearn.decomposition.NMF
```
主要参数有：
* n_components：用于指定分解后矩阵的单个维度k
* init：W矩阵和H矩阵的初始化方式，默认为'nndsvdar'

#### 实例讲解
已知Olivetti人脸数据共400个，每个数据是64*64大小。由于NMF分解得到的W矩阵相当于从原始矩阵中提取的特征，那么就可以使用NMF对400个人脸数据进行特征提取。
#### 程序设计
```python
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition
 
n_row, n_col = 2, 3          #图像展示时的排列方式
n_components = n_row * n_col #提取特征的数目
image_shape = (64, 64)       #图片大小
 
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0)) 
#加载数据并打乱顺序，RandomState(0)将随机数种子固定，既打乱了顺序又保证每次运行的结果一致
faces = dataset.data
 
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row)) #创建图片
    plt.suptitle(title, size=16)
 
    for i, comp in enumerate(images): 
        plt.subplot(n_row, n_col, i + 1)                  #绘制子图
        vmax = max(comp.max(), -comp.min())
 
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest', vmin=-vmax, vmax=vmax) 
                   #对数值归一化，并以灰度图形显示;interpolation表示色块边界的模糊程度；vmax和vmin控制亮度
        plt.xticks(())
        plt.yticks(())                                    #去除子图的坐标轴标签
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.) #调整子图位置及间隔
     
plot_gallery("First centered Olivetti faces", faces[:n_components])
 
estimators = [
    ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=6,whiten=True)),
 
    ('Non-negative components - NMF',
         decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))]
 
for name, estimator in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    print(faces.shape)
    estimator.fit(faces)
    components_ = estimator.components_
    plot_gallery(name, components_[:n_components])
 
plt.show()
```

<br>

### KMeans图像分割
#### 基本思想
利用图像的灰度、颜色、纹理、形状等特征，把图像分成若干个互不重叠的区域，并使这些特征在同一区域内呈现相似性，在不同的区域之间存在明显的差异性。然后就可以将分割的图像中具有独特性质的区域提取出来用于不同的研究。 
#### 实例讲解
目标：利用K-means聚类算法对图像像素点颜色进行聚类实现简单的图像分割。

输出：同一聚类中的点使用相同颜色标记，不同聚类颜色不同。

#本实验涉及对图片的加载和创建，所以需要PIL包，`pip install pillow`。
#### 程序设计
```python
import PIL.Image as image
from sklearn.cluster import KMeans
 
def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m,n = img.size                      #获得图片像素大小
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j)) #获得每个像素点的RGB颜色
            data.append([x/256.0,y/256.0,z/256.0]) 
    f.close()
    return data,m,n
 
imgData,row,col = loadData('F:/timg.jpg')
label = KMeans(n_clusters=4).fit_predict(imgData)
 
label = label.reshape([row,col])
pic_new = image.new("L", (row, col))    #创建一张新的图保存聚类后的结果，'L'表示灰度图
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j), int(256/(label[i][j]+1))) 
        #根据所属类别向新图中添加灰度值；第二个参数应为三元素元组，此处用int可以默认为三元素相同的元组
pic_new.save("result.jpg", "JPEG")
```
