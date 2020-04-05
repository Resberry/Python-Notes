### Github Pages
Github Pages本身就是Github提供的博客服务。在Github中创建一个特定格式的 Repository，Github Pages就会将里面的信息生成一个网页展示出来。

操作如下：

* 注册Github账号，在Github中创建一个以“用户名.github.io”结尾的Repository。（*用户名需小写*）
* 创建仓库时勾选“Initialize this repository with a README”。
* 打开网页“用户名.github.io”就可以看到README.md里的内容了。这个Repository就是用来存放博客内容的地方，也只有这个仓库里的内容，才会被网页显示出来。

### Hexo
Hexo是一个博客框架，它把本地文件里的信息生成一个网页。

操作如下：

1. 安装 Node.js

* 前往 https://nodejs.org/en/。
* 点击左侧LTS版本下载并安装。
* 打开cmd，输入 node -v得到当前版本号说明安装成功。

2. 安装 Git

* 前往 https://git-scm.com/
点击 Downloads
点击 Windows
一般情况，下载会自动开始。如果没有，就点击 click here to download manually
安装
打开 Command Prompt， 输入 git --version
得到：git version 2.15.0.windows.1

安装成功

额外说明：如果 Git –version 指令不管用，可能需要到 Environment Variable 那里添加 Path。

安装 Hexo

打开 Command Prompt
输入 npm install -g hexo-cli
回车开始安装
输入 hexo -v
得到 hexo-cli: 1.0.4 等一串数据

安装成功
