import pymysql
import pandas as pd
from crw_one_page import  filtered_news,news_contents,insert_into_db,crawling,filtering


# MySQL 연결
db = pymysql.connect(
    host="localhost",
    user="root",
    password="1514",
    database="news_db_one_page",
    charset="utf8mb4")


# 크롤링 URL
url = "https://m.blog.naver.com/sj3589/223553835059?referrerCode=1"

# 크롤링 및 필터링
news_contents = crawling(url)
filtered_news = filtering(news_contents)
# 데이터베이스에 삽입
insert_into_db(filtered_news)



sql = '''
SELECT title,time,body
FROM news

'''
# Connection 객체로부터 cursor() 메서드를 호출하여 Cursor 객체를 가져옴
cursor = db.cursor(pymysql.cursors.DictCursor)

# 실행하기
cursor.execute(sql)

# 모든 데이터를 가져온다.
datas = cursor.fetchall() 

# 사용했던 데이터베이스 관련 자원들을 닫아준다.
cursor.close()
db.close()

# 결과를 pandas 데이터프레임으로 변환
df = pd.DataFrame(datas)

print(df)