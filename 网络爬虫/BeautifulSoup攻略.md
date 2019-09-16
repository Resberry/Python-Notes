#### 提取页面信息
```python
from bs4 import BeautifulSoup
from urllib.request import urlopen
html=urlopen('https://www.baidu.com')
```
*#urlopen()返回一个类文件Response对象*

#### 先熬汤
```python
soup=BeautifulSoup(html,'html.parser')
```
*#python自带的html解析器，或者pip安装lxml解析器*

<br>

```python
soup.prettify() 
```
返回修饰过的网页源代码，呈按节点（标签）组成的树形结构。

标签以<标签名>为开始，以</标签名>为结束。

<br>

```python
tag=soup.find('参数1','参数2') 
```
（参数1为标签名，参数2为class的值）找到第一个符合的标签。

`soup.a`相当于`soup.find('a')`

<br>

```python
tag=soup.find_all('a') 
```
找到所有a标签并以列表形式返回。

由于该方法使用非常频繁，所以有一简写模式：

`tag=soup('a')`

<br>

```python
tag=soup.find(href='//www.jd.com/') 
```
根据标签的属性值寻找标签。

<br>

```python
tag.attrs
```
以字典形式返回tag标签的所有属性和其值。

<br>

```python
tag.get('href') or tag['href'] 
```
找到此tag中，属性href对应的值

<br>

```python
tag.contents
```
以列表形式返回tag的子节点

<br>

```python
tag.parent
```
返回tag的父节点

<br>

`tag.children` 或 `tag.descendants` 或 `tag.parents`

子节点/孙节点/先辈节点生成器，不能直接输出，仅用于循环遍历，例如：
```python
for child in tag.children: 
    print(child)
```

#### 找文本内容：
若tag只有一个子节点，用
```python
tag.string or tag.get_text()
```
提取文本。

<br>

若tag含有多个子、孙节点，且每个节点都有文本，则用
```python
for string in tag.strings: #一个迭代器
    print(repr(string))
```
