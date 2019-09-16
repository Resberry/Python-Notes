打开Sublime Text3，依次点击Tools—Build System—New Build System，

在代码中加入以下段落：

```python
{
"cmd": ["D:\\Software\\Python37\\python.exe","-u","$file"],
"file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
"selector": "source.python",
"env": { "PYTHONIOENCODING": "utf8" },
}
```

保存为python3

*#注意其中python.exe的路径为自己电脑中的安装路径。*
