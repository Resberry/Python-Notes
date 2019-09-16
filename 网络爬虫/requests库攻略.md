#### 提取并查看网页源代码
```python
import requests
url='......'
r=requests.get(url) #r为Response类型
```
*#此时response可作为BeautifulSoup参数。*

##### 查看网页源代码
`r.content.decode()` 或 `r.text`

*#若r.text输出乱码，需要将r.encoding改为正确的编码。*

##### 查看http中的headers

`r.headers`

##### 从headers中猜测的响应的内容的编码方式
`r.encoding`

##### 从内容中分析的编码方式
`r.apparent_encoding`

##### 响应内容（即网页源代码）的二进制形式
`r.content`

##### 请求的返回状态，200为成功，404为失败
`r.status_code`

##### 如果状态不是200，则引发HTTPError异常
`r.raise_for_status()`

##### 请求网页的通用代码框架：
```python
def getHTMLText(url):
    try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
    except:
        return "产生异常"           
```
        
#### 通过requests添加headers
```python
import requests
url='https://......'
header={'user-agent'='......'}
response=requests.get(url,headers=header)
```

#### 设定代理ip
```python
import requests
url='https://......'
pxs={'https':'https://10.10.10.1:4321'}
response=requests.get(url,proxies=pxs)
```

#### 爬取图片并保存在本地
创建图片文件并以二进制形式打开，从网页提取图片网址url。
```python
f.write(requests.get(url).content)
```
