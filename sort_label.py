import pandas as pd
import re
label = [
'environmental','sfda','enforce_main_type_13','land','enforce_main_type_22','enforce_main_type_56','enforce_main_type_19','enforce_main_type_11','enforce_main_type_14','enforce_main_type_12','enforce_main_type_25',
'enforce_main_type_8','enforce_main_type_29','enforce_main_type_20','enforce_main_type_47','enforce_main_type_38','enforce_main_type_6','enforce_main_type_31','enforce_main_type_30','enforce_main_type_7',
'enforce_main_type_10','enforce_main_type_18','enforce_main_type_36','enforce_main_type_15','enforce_main_type_78','enforce_main_type_16','enforce_main_type_49','enforce_main_type_57',
'enforce_main_type_5','enforce_main_type_37','enforce_main_type_75','enforce_main_type_89','enforce_main_type_9','enforce_main_type_58','enforce_main_type_43','enforce_main_type_32','enforce_main_type_41','enforce_main_type_28',
'enforce_main_type_23','other','enforce_main_type_42','enforce_main_type_61','enforce_main_type_77']
print(len(label))


tar_list = [i.strip() for i in open('C:/Users/ASUS/Desktop/label_bj.txt', 'r', encoding='utf8')]#strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
g = open('C:/Users/ASUS/Desktop/label_bj1.txt',"w",encoding = "utf8")
tar_list_ = []
count = 0
for i in tar_list:
    if i == "environmental":
        g.write('环境')
        g.write('\n')
        count+=1
        if count % 1000 == 0:
            print(count)
    elif i == "sfda":
        g.write("食品药品监督管理局")
        g.write('\n')
        count += 1
        if count % 1000 == 0:
            print(count)
    else:
        g.write("其他")
        g.write('\n')
        count += 1
        if count % 1000 == 0:
            print(count)



