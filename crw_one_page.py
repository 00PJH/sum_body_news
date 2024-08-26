from bs4 import BeautifulSoup
import requests
import pymysql
import re

# MySQL 연결 설정
db = pymysql.connect(
    host="localhost",
    user="root",
    password="1514",
    database="news_db_one_page",
    charset="utf8mb4"
)

# 크롤링 함수
def crawling(url):
    response = requests.get(url)
    state = response.status_code
    data = []

    if state == 200:
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')
        data = soup.find_all(class_='se-fs-fs16 se-ff-system')
    else:
        print(f"state code: {state}")

    return data

# 필터링 함수: newsroom, title, time, body 추출
def filtering(contents):
    newsData = []
    body = ""
    
    # 뉴스 데이터 패턴: [언론사] 제목 (날짜)
    pattern = r"\[(.*?)\] (.*?) \((.+)\)$"

    for content in contents:
        text = content.text.strip()
        
        # 특수 문자 포함된 불필요한 문장 필터링 (예: ★로 시작하는 문장)
        if '★' in text :
            continue  # 불필요한 내용이면 넘어감

        # 제목 및 언론사, 시간 추출
        match = re.search(pattern, text)

        if match:
            newsroom = match.group(1)
            title = match.group(2)
            time = match.group(3)
            continue
        
        # 기사 본문이 끝났는지 확인 후 저장
        if text.startswith('-') and text.endswith('-'):
            newsDict = {
                "newsroom": newsroom,
                "title": title,
                "time": time,
                "body": body
            }
            newsData.append(newsDict)
            body = ""  # 본문 초기화
        else:
            body += text  # 본문 이어쓰기

    return newsData

# 데이터베이스에 저장하는 함수
def insert_into_db(data):
    cursor = db.cursor()

    for news in data:
        newsroom = news['newsroom']
        title = news['title']
        time = news['time']
        body = news['body']

        # 중복 확인 쿼리
        select_query = "SELECT COUNT(*) FROM news WHERE title = %s"
        cursor.execute(select_query, (title,))
        result = cursor.fetchone()

        if result[0] == 0:
            # 중복되지 않은 경우 삽입
            insert_query = """
                INSERT INTO news (newsroom, title, time, body) 
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (newsroom, title, time, body))
            db.commit()
            print(f"Record inserted: {title}")
        else:
            print(f"Duplicate entry found for title: {title}")

    cursor.close()

# 크롤링 URL
url = "https://m.blog.naver.com/sj3589/223560703851?referrerCode=1"

# 크롤링 및 필터링
news_contents = crawling(url)
filtered_news = filtering(news_contents)

# 데이터베이스에 삽입
insert_into_db(filtered_news)

# 데이터베이스 연결 종료
db.close()


# CREATE DATABASE news_db_one_page;

# USE news_db_one_page;

# CREATE TABLE news (
#     title VARCHAR(255) PRIMARY KEY,
#     newsroom VARCHAR(255),
#     time VARCHAR(255),
#     body TEXT
# );
# 이건 테이블 정의할 때

