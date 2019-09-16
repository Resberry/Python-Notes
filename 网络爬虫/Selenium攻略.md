#### 安装Selenium

`pip install Selenium`

Selenium是一个强大的网络数据采集工具，是一个在WebDriver上调用的API。

WebDriver有点儿像可以加载网站的浏览器，但是它也可以像BeautifulSoup对象一样用来查找页面元素，与页面上的元素进行交互（发送文本、点击等），以及执行其他动作来运行网络爬虫。

Selenium自己不带浏览器，它需要与第三方浏览器结合在一起使用。

#### 元素定位
```python
find_element_by_id
find_element_by_name
find_element_by_xpath
find_element_by_link_text
find_element_by_partial_link_text
find_element_by_tag_name
find_element_by_class_name
find_element_by_css_selector
get_attribute #获取该元素下的某属性值
text          #提取文本
#除按id查找外，将element改为elements即会找到所有符合条件的元素，然后返回一个列表
```

#### PhantomJS
去官网下载，然后把exe文件拷到Python安装目录的Scripts文件夹下。

由于新版Selenium已不支持PhantomJS，所以只能使用3.8.0及以下版本的Selenium。
```python
from selenium import webdriver
url="https://www.baidu.com"
driver=webdriver.PhantomJS()
driver.get(url)
driver.save_screenshot('info.png')        #截全屏
print(driver.title)                       #查看当前浏览器标题
text=browser.find_element_by_id('kw')     #找到搜索框
text.clear()                              #清空搜索框
text.send_keys('python')                  #填写搜索框的文字
button = browser.find_element_by_id('su') #找到搜索提交按钮
button.click()                            #点击按钮，提交搜索请求
#button.submit()
```

#### 基本注意事项
有些源代码可能有若干frame区，初始情况下frame内部的元素无法被定位到。如果想进入指定的frame，可以用
```python
driver.switch_to.frame(参数)       #参数可以为序号（0是第一位），也可以为frame的name。
driver.switch_to.default_content() #回到主文档
driver.switch_to.parent_frame()    #回到上一frame
```
是否添加headers、使用不同的headers可能会对请求到的源代码有影响。

`find_element_by_class_name`中的class参数中不能有空格，如果有可以用xpath定位。

对于动态id或class应该如何定位：
1. 根据其他属性定位。
2. 根据相关关系或排序定位。
3. 用xpath特殊定位：
```python
driver.find_element_by_xpath("//div[contains(@id, 'btn-attention')]")
driver.find_element_by_xpath("//div[starts-with(@id, 'btn-attention')]")
driver.find_element_by_xpath("//div[ends-with(@id, 'btn-attention')]")  
#contains(a, b) 如果a中含有字符串b，则返回true，否则返回false
#starts-with(a, b) 如果a是以字符串b开头，返回true，否则返回false
#ends-with(a, b) 如果a是以字符串b结尾，返回true，否则返回false
```

#### 三种等待方式
有些网页采用动态加载方式，所以需要等待其内容加载完毕，否则会出现查找不到元素等问题。
1. 强制等待
```python
import time
time.sleep(3) #代码运行到此处后强制休眠3秒
```
2. 隐性等待
```python
implicitly_wait(3)
#效果覆盖全局，在查找每个元素前都会等待至多3秒，3秒以内若js加载完毕会立即往下运行。
#弊端：程序会一直等待整个页面加载完成才会执行下一步，但有时候页面想要的元素早就在加载完成了，但是因为个别js之类的东西特别慢，仍需继续等待。
```
可以在不需要等待的语句前将隐性等待调为0，过后马上调回来。

3. 显性等待

`WebDriverWait`，配合该类的`until()`和`until_not()`方法，就能够根据判断条件而进行灵活地等待了。它主要的意思就是：程序每隔x秒看一眼，如果条件成立了，则执行下一步，否则继续等待，直到超过设置的最长时间，然后抛出TimeoutException。
 ```python
WebDriverWait(driver, 超时时长, 调用频率, 忽略异常).until(可执行方法, 超时时返回的信息)
```
其中的可执行方法，一般使用`expected_conditions`模块中的各种条件。
`expected_conditions`是Selenium的一个模块，其中包含一系列可用于判断的条件：
```python
title_is
title_contains
#验证title，验证传入的参数title是否等于或包含于driver.title
presence_of_element_located
presence_of_all_elements_located
#验证元素是否出现，传入的参数都是元组类型的locator，如(By.ID, 'kw')
#第一个语句只要一个符合条件的元素加载出来就通过；第二个必须所有符合条件的元素都加载出来才行
visibility_of_element_locatedin
visibility_of_element_located
visibility_of
#验证元素是否可见，前两个传入参数是元组类型的locator，第三个传入WebElement。第一个和第三个其实质是一样的
text_to_be_present_in_element
text_to_be_present_in_element_value
#判断某段文本是否出现在某元素中，一个判断元素的text，一个判断元素的value
frame_to_be_available_and_switch_to_it
#判断frame是否可切入，可传入locator元组或者直接传入定位方式：id、name、index或WebElement
alert_is_present
#判断是否有alert出现
element_to_be_clickable
#判断元素是否可点击，传入locator
element_to_be_selected
element_located_to_be_selected
element_selection_state_to_be
element_located_selection_state_to_be
#判断元素是否被选中，第一个条件传入WebElement对象；第二个传入locator元组；第三个传入WebElement对象以及状态，相等返回True，否则返回False；第四个传入locator以及状态，相等返回True，否则返回False
staleness_of
#判断一个元素是否仍在DOM中，传入WebElement对象，可以判断页面是否刷新了
```

#### 一个实例
```python
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
driver = webdriver.PhantomJS()
driver.implicitly_wait(10) # 隐性等待和显性等待可以同时用，但要注意：等待的最长时间取两者之中的大者
driver.get('https://huilansame.github.io')
locator = (By.LINK_TEXT, 'CSDN')
try:
  WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
  print(driver.find_element_by_link_text('CSDN').get_attribute('href')
finally:
  driver.close()
```

#### 其他浏览器
##### Chrome
首先查看电脑中chrome浏览器的版本，根据chromedriver与chrome版本映射表，进入http://npm.taobao.org/mirrors/chromedriver/下载对应的chromedriver.exe，然后将其分别放入chrome浏览器以及python的安装目录下。
```python
from selenium import webdriver
url="https://www.baidu.com"
options = webdriver.ChromeOptions()      #准备配置chrome这个类
options.add_argument('user-agent="Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19",--headless') #设置headers
options.add_argument('--headless')       #启动无头模式
options.add_argument('--disable-gpu')    #无头模式下加入此参数来规避bug
driver=webdriver.Chrome(options=options) #初始化实例，括号中的内容即为将上述配置封装起来
```

##### Firefox
https://github.com/mozilla/geckodriver/releases下载对应版本的geckodriver.exe，放入Firefox浏览器以及python的安装目录下。
```python
from selenium import webdriver
url="https://www.baidu.com"
options = webdriver.FirefoxOptions() #准备配置Firefox
options.add_argument('-headless') #启动无头模式
driver = webdriver.Firefox(options=options) #初始化实例
```

#### 基本操作
```python
driver.get(url) #发出请求。WebDriver会等待页面完全加载完成之后才会返回，即程序会等待页面的所有内容加载完成，JS渲染完毕之后才继续往下执行。所以，我们可以得到JS渲染之后的页面源码。
html = driver.page_source                 #当前区域的源代码
driver.save_screenshot('路径+名称+后缀名') #生成页面快照并保存，仅限当前窗口
driver.close() #关闭当前页面
#driver.quit() #关闭浏览器
print(html)
```

#### 可能出现的问题
Chrome和Firefox下运行完成后，python都不会显示Finished时间。

Chrome下元素定位时可能会无事发生，即不会显示任何结果。

Firefox下运行代码时，偶尔会出现`Message:Connection refused`，可能与Selenium、Firefox、geckodriver版本匹配有关，尚未确定解决方案。











