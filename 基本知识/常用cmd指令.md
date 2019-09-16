#### 更新pip版本
```
python -m pip install -U pip
```

#### 设置超时时间
```
pip --default-timeout=1000 -U pip
```

#### 更换pip下载源
```
pip install 插件名 -i 新下载源网址 #如https://pypi.douban.com/simple/
```

#### 用pip进行安装卸载
```
pip install 工具名       #安装
pip install 工具名==版本 #安装该工具的指定版本
pip uninstall 工具名     #卸载
pip show 工具名          #显示该工具的版本等详细信息
```

#### 进入目录
直接输入盘符加冒号进入该盘，然后输入cd 路径，进入该路径。*#分隔符为反斜杠*

`cd..`：返回上一级。

`cd \`*#正反斜杠均可*：返回当前盘符。

`tree`：显示当前目录下的树形结构，只显示文件夹。

`tree /f`：显示当前目录下的树形结构，包括文件夹和文件。

#### 退出cmd
`ctrl+Z+Enter`

#### 查看python版本
`python --version`
