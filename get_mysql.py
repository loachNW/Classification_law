import pymysql


db = pymysql.connect(host='172.16.11.34', user='viewuser', passwd='viewuser', db="intelligent", port=6033, charset='utf8')
cur = db.cursor()
cur.execute("select industry_name, industry_show_name from industry")
data = cur.fetchall() #所有
data = list(data)
for item in data:
    print(item)
db.close()
