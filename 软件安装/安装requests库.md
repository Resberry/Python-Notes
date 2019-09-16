在cmd中输入

`pip install requests`

<br>

如果出现红字报错（read timed out），可输入

`pip --default-timeout=1000 -U pip`

修改超时时间。

<br>

如果再次出现黄字报错且无法正常安装，可以更换安装源进行安装：

`pip install requests -i https://pypi.doubanio.com/simple`

（以上为豆瓣的pip源，我国还有很多其他pip源。）
