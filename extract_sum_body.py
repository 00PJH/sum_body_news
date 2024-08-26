import pymysql
import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from test_abssumm import newSum

# MySQL 연결 설정
db = pymysql.connect(
    host="localhost",
    user="",  # 자신의 db 이름
    password="", # 자신의 db 비밀먼호
    database="news_db_one_page",
    charset="utf8mb4"
)


  # news_sum_body 테이블 생성 (없을 경우)
create_table_query = """
    CREATE TABLE IF NOT EXISTS news_sum_body (
        title VARCHAR(255) PRIMARY KEY,
        sum_body TEXT
        )
    """

extract_title_body_query = """
    SELECT title, body
    from news
"""

save_title_sum_body_query = """

"""

# db 연결
cursor = db.cursor(pymysql.cursors.DictCursor)
# create table query  실행
cursor.execute(create_table_query)

# extract_title_body_query  실행
cursor.execute(extract_title_body_query)

# 모든 데이터를 가져온다.
datas = cursor.fetchall() 


# 결과를 pandas 데이터프레임으로 변환
df = pd.DataFrame(datas)

# 요약문 저장할 리스트 생성
sum_body_list = []


# Load Model and Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

# 요약문 리스트에 저장
for text in df['body']:
    sum_body_list.append(newSum(text))



for i in range(len(sum_body_list)):
    insert_query = """
            INSERT INTO news_sum_body (title, sum_body) 
            VALUES (%s, %s)
    """

    cursor.execute(insert_query, (df['title'][i], sum_body_list[i]))
    db.commit() 


# 커서 및 DB 연결 종료
cursor.close()
db.close()