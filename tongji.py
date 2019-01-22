import pandas as pd

tar_list = [i.strip() for i in open('label2.txt', 'r', encoding='utf8')]#strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
all_words_sr = pd.Series(tar_list)#Series创建一个系列
all_words_counts = all_words_sr.value_counts()#pandas 的 value_counts() 函数可以对Series里面的每个值进行计数并且排序。
print(all_words_counts)
pass
