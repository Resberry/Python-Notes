首先需要先pip安装lxml解析器。

然后依然pip安装scrapy，但进行到安装Twisted时会报错。因为默认的Twisted与高版本python不兼容。

解决办法：去https://www.lfd.uci.edu/~gohlke/pythonlibs/ *#python扩展包的非官方Windows二进制文件* 下载对应当前python版本的Twisted.whl文件，将其放入python目录下的Scripts文件夹。

打开cmd，将路径转到Scripts文件夹，

`pip install Twisted xxx.whl`

*#即下载的文件的名称*，即可成功安装。

然后再

`pip install scrapy`

想运行scrapy爬虫，还需安装pypiwin32，直接pip即可。
