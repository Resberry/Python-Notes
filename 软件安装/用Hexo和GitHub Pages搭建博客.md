## Github Pages
Github Pages 本身就是 Github 提供的博客服务。在 Github 中创建一个特定格式的 Repository，Github Pages 就会将里面的信息生成一个网页展示出来。

操作如下：

* 注册 Github 账号，在 Github 中创建一个名为 resberry.github.io的Repository （*resberry 为你的 github 账户用户名且必须小写*）。
* 创建仓库时勾选 Initialize this repository with a README。
* 打开网页 resberry.github.io 就可以看到 README.md 里的内容了。这个 Repository 就是用来存放博客内容的地方，也只有这个仓库里的内容，才会被网页显示出来。

## Hexo
Hexo 是一个博客框架，它把本地文件里的信息生成一个网页。

操作如下：
1. 安装 Node.js
* 前往 https://nodejs.org/en/ 。
* 点击左侧 LTS 版本下载并安装。
* 打开 cmd，输入`node -v`得到当前版本号说明安装成功。
2. 安装 Git
* 前往 https://git-scm.com/ 。
* 点击 Downloads，点击 Windows。若未开始下载就点击 click here to download manually 。
* 打开 cmd，输入`git --version`得到当前版本号说明安装成功（*若 git 指令无效则需重启或去环境变量添加 path *）。
3. 安装 Hexo
* 打开 cmd ，输入`npm install -g hexo-cli`。
* 输入`hexo -v`得到 hexo-cli 等一串数据说明安装成功。

4. 创建本地博客
* 在本地创建文件夹，如 D 盘创建文件夹 blog 。
* 鼠标右键 blog，选择 Git Bash Here。打开后，所在位置就是 blog 文件夹的位置 /d/blog。
* 输入`hexo init`将 blog 文件夹初始化成一个博客文件夹。
* 输入`npm install`安装依赖包。
* 输入`hexo g`生成网页（generate）。由于我们还没创建任何博客，生成的网页会展示 Hexo 自带的 Hello World 博客。
* 输入`hexo s`将生成的网页放在本地服务器（server）。
* 浏览器里输入 http://localhost:4000/ 即可预览网页。回到 Git Bash，按 Ctrl+C 结束预览。

## 将本地 Hexo 博客部署在 Github 上
我们现在已经有了本地博客，和一个能托管这些资料的线上仓库。只要把本地博客部署（deploy）在我们的 Github 对应的 Repository 就可以了。

操作如下：

* 获取 Github 对应的 Repository 的链接。登陆 Github，进入到 resberry.github.io，点击 Clone or download，复制 URL 待用。（https://github.com/Resberry/resberry.github.io.git）

* 修改博客的配置文件。用 IDE 打开配置文件 /d/blog/_config.yml，找到 #Deployment填入以下内容：
```
deploy:  
	  type: git  
	  repository: https://github.com/Resberry/resberry.github.io.git
	  branch: master
```
* 回到 Git Bash，输入`npm install hexo-deployer-git --save`安装 hexo-deployer-git。
* 输入`git config --global user.name "github用户名"`和`git config --global user.email  "github邮箱名"`登录个人信息。
* 输入`hexo d`得到`INFO Deploy done: git`即为部署成功，之前我们创建的 README.md 会被自动覆盖掉（*若报错可以删掉路径下的 .deploy_git 文件夹重试*）。


## 发布一篇博客

* 在 Git Bash 里，所在路径还是 /d/blog。输入`hexo new "日志名"`，在 D:\blog\source\_posts 路径下会出现一个 日志名 .md 的文件，编辑此文件并保存。也可去该路径手动创建 .md 文件
* 回到 Git Bash，输入`hexo g`和`hexo d`部署（*或直接输入`hexo d -g`*）。
