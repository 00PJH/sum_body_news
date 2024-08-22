from transformers import BertModel, BertTokenizer
import pymysql
import pandas as pd
from test_extsumm import split_into_sentences,sum_body 

# MySQL 연결 설정
db = pymysql.connect(
    host="localhost",
    user="root", # 본인 db 이름
    password="1514", # 본인 db 비밀번호
    database="news_db", # 해당 db
    charset="utf8mb4"
)

sql = '''
SELECT title, time, body
FROM news
'''

insert_query = """
INSERT INTO news (title, time, body) 
VALUES (%s, %s, %s, %s)
"""

# Connection 객체로부터 cursor() 메서드를 호출하여 Cursor 객체를 가져옴
cursor = db.cursor(pymysql.cursors.DictCursor)

# 실행하기
cursor.execute(sql)

# 모든 데이터를 가져온다.
datas = cursor.fetchall() 

# 결과를 pandas 데이터프레임으로 변환
df = pd.DataFrame(datas)



# KoBERT 모델 및 토크나이저 로드
model_name = 'monologg/kobert'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


result_sum_body = []
for text in df['body'] :
    result_sum_body.append(sum_body(text))



# 사용했던 데이터베이스 관련 자원들을 닫아준다.
cursor.close()
db.close()