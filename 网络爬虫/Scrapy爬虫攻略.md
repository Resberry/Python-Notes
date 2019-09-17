首先打开cmd，进入一个路径，该路径即为存放爬虫框架的位置。

创建一个scrapy工程：

`scrapy startproject 工程名`


然后进入工程目录：

`cd 工程名`


创建爬虫：

`scrapy genspider 爬虫名 网页域名`


#### 然后进入spiders文件夹打开爬虫.py文件，进行定制编辑
parse函数功能是接收response，解析爬取的内容，提取所需的内容，并返回新的需要爬取的请求。

*#有些html源代码实际结构与F12中的不一样，爬取前可以先检查一下。*

*#这里如果用beautifulsoup筛选信息，注意第一个参数应为response.body。*

*#该函数的返回值类型只能为Request,BaseItem,dict or None或者由它们构成的列表。*

如果爬虫需要读取结构不同的网页，那么一个parse函数就不够用了，我们可以模拟出点击链接进入网页的效果，即使用scrapy.Request以及回调函数：
写好回调函数后，在parse函数下，使用语句
```python
return/yield scrapy.Request(url，callback=self.函数名) #在循环中使用yield
```

该语句作用是对需要进入的新的url发送请求，然后将response作为参数返回callback的函数。回调的函数与parse函数一样，都在一个类中，参数均为self和response。

*#注意，scrapy异步处理Request请求，即scrapy发送请求后，不会等待这个请求的响应，而会同时发送其它请求或做别的事情。*

#### 编辑pipelines.py
功能是处理爬虫获取的内容（如打印、写入文件）。

*#该文件中的函数最后有一个return item，功能是运行爬虫后，会在cmd上显示出你筛选的信息。*

如果一个工程中有多个爬虫，可以加一个判断句，根据爬虫名字spider.name的不同进行不同操作。

*#这种情况下，若要运行某个爬虫，需要保证所有爬虫文件都无语法错误。*

此文件中自带一个类，也可以创建新的类进行不同操作，但注意需要在settings.py中添加相应配置。

注意如果要使用os.getcwd()的话，返回的是运行该语句时所在的路径，而不一定是这条语句所在文件的路径。

如果要爬图片，注意获取src后要补上https://。

#### 编辑items.py
只需将你需要获取的信息的变量名称 *#即存放在item字典中的关键词都加入类中即可*，形如key=scrapy.Field()

*#这里的目的是为了构造一个类作为爬虫.py中item的类型，类中包括了item的所有分量。然后在爬虫.py中需要先*
```python
from 工程名.items import 工程名Item #即类的名字
```
*然后*
```python
item=工程名Item()
```
*如果数据量小也可以直接将item设为字典，此时就可以无视items.py了。*

#### 编辑settings.py
即爬虫的配置文件。

首先需要加入
```python
ITEM_PIPELINES={'工程名.pipelines.工程名Pipelines#即pipelines.py文件中自带的类':300,}
```
*#否则pipelines.py无法正常运行。*

*#如果pipelines.py中还有别的类，在字典里继续添加即可，数字可以依次为400、500，数字小的先执行。*

最后在cmd中该工程目录下运行scrapy crawl 爬虫名，即可启动爬虫。

#### 反爬应对措施
如果运行爬虫，得到403报错，或者提示
```python
twisted.python.failure.Failure twisted.internet.error.ConnectionDone: Connection
```
说明所爬网页可能有headers反爬手段，那么需要在此文件中添加一行
```python
USER_ AGENT='任意一个浏览器header'。
```

如果想无视robots.txt协议，输入
```python
ROBOTSTXT_OBEY = False
```
即可。

503 Service Unavailable：

可能是网站有如下反爬机制：

1. 时间间隔封锁：当请求过于频繁时禁止访问，可以在settings.py中加入
```python
DOWNLOAD_DELAY=1
```
设置下载延迟。

2. Cookie封锁：cookies是用来确定用户身份的一长串数据，若网站对cookie进行简单检验，可以设置
```python
COOKIES_ENABLED=False
```
来禁用cookie。

#### 常用http状态码
200：一切正常。

30x：重定向，例如用户向一个旧的url发送了请求，系统会自动定位到正确的url。

401：当前请求需要用户验证，用户的请求未提供正确的认证证书。

403：服务器已理解请求，但是拒绝执行。

404：请求失败，请求所希望得到的资源未被在服务器上发现。

503：由于临时的服务器维护或过载，服务器当前无法处理请求。这个状况是临时的。
