```python
from scrapy.selector import Selector
from urllib.request import urlopen
```
*#这里举例的是单独使用xpath的情况，如果是在scrapy框架中，不用import，直接在parse函数中response.xpath即可。*
```python
url='......'
html=urlopen(url)
te=html.read()
```
*#Selector函数的text接受Unicode、str和bytes型数据。*

*#也可以自己创建xml文件，即打开txt，按html源代码格式书写文本，将扩展名改为xml。*
```python
tag=Selector(text=te).xpath('......').extract()
```
*#xpath会找到所有符合条件的元素，然后返回一个列表。*

*#一般来说，返回的列表中没有重复元素。*

这里extract()是将内容提取成字符串的列表，如果不加extract()，则返回内容的格式依旧是Selector型的列表，那么进行嵌套时可以直接加.xpath。
举例：
```python
tag1=Selector(text=te).xpath('//A')[0]
tag2=tag1.xpath('B/text())
```
*#这里B前面不加/，代表tag1的子标签。*

嵌套：text可以进行嵌套，嵌套的标签中隐含着其在总文本中的前两个标签。 *#比如<html>和<body>*

下面重点讲解xpath括号中内容：

括号中可以直接以标签开头，但此标签必须为当前标签的子标签。

<br>

.：

选取当前标签。可以单独使用'.'，查看当前标签的内容。 *#包括当前标签在总文本中的前两层标签。*

<br>

..：

.../A/..找到A标签的父标签。

<br>

/：

...A/B，找到A标签下的所有B子标签。注意一定是子标签，因为/无法跨代。 *#若路径以/开头，则一定是绝对路径，即从最外层标签开始算起。*

<br>

//：

...A//B，找到A标签下的所有B标签，无论B标签在什么位置。

<br>

@：

//A[@class="li"]，找到所有含有class="li"属性的A标签；//A/@class，找到所有A标签的class属性的值。

<br>

[ ]：

//A/B[1]，选取所有A标签，分别找到它们的第一个B子标签；同理[last()]为分别找到最后一个子标签；[position()<3]为分别找到前两个子标签。

<br>

text()：

...A/text()，找到A元素的文本。

<br>

*：

*，匹配任何标签；@*，匹配任何属性。

<br>

|：

可以一次选取若干个路径。 *#均包含在一个字符串内*

//A|//B，找到所有A标签和所有B标签。
