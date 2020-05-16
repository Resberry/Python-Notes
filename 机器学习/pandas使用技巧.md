输出数据时，中文表头未和数据对齐，使用以下方法解决：
```python
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
```
在jupyter中，若直接使用df.head()等函数输出数据时，便可得到美观的数据格式；若加上print()，则会得到朴素的数据格式。
