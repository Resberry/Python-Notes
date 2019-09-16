#### 自定义MIddleWares中间件
在与settings.py的同级目录建立文件夹：middlewares，接着，在文件夹下创建：\_\_init\_\_.py，这样可以让Python认为这个文件是一个可以导入的包。

然后，我们开始写中间件：customUserAgent.py

*#功能：每次爬虫随机选择一个user-agent*
```python
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
import random
agents = ['Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv,2.0.1) Gecko/20100101 Firefox/4.0.1',
'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)']
class RandomUserAgent(UserAgentMiddleware):
    def process_request(self,request,spider):
        '''定义下载中间件，必须要写这个函数，这是scrapy数据流转的一个环节。'''
        ua = random.choice(agents)
        request.headers.setdefault('User-agent',ua)
```

当然，我们需要在settings.py里激活我们的下载中间件：

注意，需要scrapy自身的user-agent中间件关闭！
```python
DOWNLOADER_MIDDLEWARES = {
    '工程名.middlewares.coustomUserAgent.RandomUserAgent': 20,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware':None，}
```

接着，创建我们的中间件：coustomProxy.py

*#功能：爬虫每次访问页面的时候都会换一个ip*
```python
proxies = [
    '89.36.215.72:1189',
    '94.177.203.123:1189',
    '110.73.11.227:8123',
    '180.183.176.189:8080',
    '109.62.247.81:8080']
import random
class RandomProxy(object):
    def process_request(self,request,spider):
        proxy = random.choice(proxies)
        request.meta['proxy'] = 'http://{}'.format(proxy)
```

最后，设置settings.py，注意，这里和上面不一样，不能关闭scrapy本身的代理中间件，只需要让自己写的在官方之前执行就成。
```python
DOWNLOADER_MIDDLEWARES = {
    '工程名.middlewares.coustomProxy.RandomProxy':10,
    '工程名.middlewares.coustomUserAgent.RandomUserAgent': 20,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware':None,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware':100,}
```
