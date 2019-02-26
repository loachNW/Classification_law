import pymongo
import re
import jieba
def cut_word(text):
    restr = '[0-9\s+\.\!\/_,$%^*();?:\-<>《》【】+\"\']+|[+——！，；。？：、~@#￥%……&*（）]+'
    resu = text.replace('|', '').replace('&nbsp;', '').replace('ldquo', '').replace('rdquo',
                                                                                    '').replace(
        'lsquo', '').replace('rsquo', '').replace('“', '').replace('”', '').replace('〔', '').replace('〕', '')
    resu = re.split(r'\s+', resu)
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('', ''.join(resu))
    line = re.sub(restr, '', dd)
    seg_list = jieba.lcut(line)
    return seg_list

myclient = pymongo.MongoClient("mongodb://intelligent:intelligent@172.16.11.199:27017/operation")
mydb = myclient['operation']
mycol = mydb['clues']
count = 0
f = open('C:/Users/ASUS/Desktop/train_test.txt',"w",encoding = "utf8")
g = open('C:/Users/ASUS/Desktop/label_bj.txt',"w",encoding = "utf8")



for x in mycol.find({ 'passFlag': 1,"clue.createDate": { "$gt":1547827200000}}):
    try:
        content = x.get('clue').get('content')
        title = x.get('clue').get('caseName')
        train = title + content
        label = x.get('operation').get("intelligent").get('recommendedIndustries')[0].get('name')
        if len(label) != 0:
            labels = []
            labels.append(label)
            word = cut_word(content)
            if len(word) > 20:
                f.write(" ".join(word) + "\n")
                g.write(" ".join(labels) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(count)
    except:
        print(x.get('clue').get('uniqid'))
f.close()
g.close()