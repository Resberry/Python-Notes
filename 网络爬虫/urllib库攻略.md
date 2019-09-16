```python
html=urlopen(url) #有些url中含有/#/，若去掉则提取到的源代码可能不同
print(html.read())
```
`html.read()`的类型为bytes，会输出十六进制乱码，若想查看正常格式的源代码，可以改为
```python
print(html.read().decode('utf-8'))
```
*#`decode('编码名')`表示用该编码方式将bytes解码成Unicode字符串。若括号内编码与该bytes本身编码方式不同，则无法正常解码。*

*#`encode('编码名')`表示用该编码方式将字符串编码为bytes。*

*#decode括号可为空。*

<br>

通过urllib添加headers：
```python
from urllib.request import urlopen,Request
header={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
req=Request(url,headers=header)
html=urlopen(req)
```
*#此时html可作为BeautifulSoup参数。*


