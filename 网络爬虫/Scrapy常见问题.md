若运行日志出现
```python
[scrapy.spidermiddlewares.offsite] DEBUG: Filtered offsite request
```
报错，说明请求的url与allowed_domains域名不一致，被过滤掉了，若想避免，可以在Request函数中设置参数`dont_filter = True`。
