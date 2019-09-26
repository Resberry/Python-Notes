由于官方网站被墙，无法在软件里自动安装Package Control。

所以将下载好的"Package Control.sublime-package"文件，放在软件中Preferences-Browse Packages目录的上一级中的Installed Packages文件夹中，即可成功安装。

这时打开从Preferences打开Package Control下载插件，会报错安装失败，还是因为网络问题。

所以从网上下载channel_v3.json文件，放在本地任意目录中，在软件中打开Preferences-Package Settings-Package Control-Settings-User，在程序第一段的channel列表中加入一行"上述文件的路径"，注意加逗号，然后保存。

这时即可下载插件。

<br>

为配置可交互环境，我们打开Package Control，输入Install Package回车，在新出现的输入框里输入SublimeREPL，点击即可安装。

使用时点击Tools-SublimeREPLsublimerepl-Python-Python-RUN current file即可为当前代码进行可交互编译。*#编译前一定要先手动保存*
