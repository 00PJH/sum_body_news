from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# KoBERT 모델 및 토크나이저 로드
model_name = 'monologg/kobert'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 문장 분리 함수
def split_into_sentences(text):
    # 간단한 문장 분리 로직
    sentences = text.split('.')
    return [sentence.strip() + '.' for sentence in sentences if sentence.strip()]


# # 문서 예제
document = '''
★2024년 6월18일 화요일 방산뉴스 브리핑★
​
1) [국방일보] 수소 모빌리티 군사적 활용으로 미래전장 선도한다  (2024. 06. 17   17:23)
​
미래 기계화부대의 완벽한 지속지원능력을 보장할 수소 모빌리티의 군 적용 방안을 모색하기 위해 민·군 전문가가 머리를 맞댔다.
육군7기동군단은 17일 군단 대회의실에서 박재열(중장) 군단장 주관으로 ‘미래 수소 모빌리티 군 적용 민·군 합동 세미나’를 개최했다. 특히 이번 세미나는 군 정책 부서나 연구기관이 아닌 야전부대가 미래 전투수행 방안에 관한 고민을 토대로 주도적으로 추진했다는 면에서 더욱 눈길을 끈다.
세미나에는 국방부·육군본부·지상작전사령부·군수사령부·국군수송사령부·미래혁신연구센터·전력지원체계사업단·방위사업청 등 군 관계자는 물론 산업통상자원부·현대자동차·기아자동차·현대로템·한양대·전북대·한국자동차연구원 등 정부 기관과 산·학·연 전문가 등 100여 명이 참석했다. 이들은 미래 수소연료체계의 군사적 활용 가능성과 민·군 합동 발전 방안을 논의했다.
세미나는 △작전환경 변화에 따른 육군 미래 에너지 공급체계 소개 △수소연료체계 활용 개발 장비 소개 △수소 모빌리티 군 적용 방안 △현장 토의 및 장비 시연 순서로 진행됐다. 참석자들은 수소 에너지가 전투 발전에 미치는 효과에 공감대를 형성하고 향후 민군 협력체계 토대를 구축하기 위한 방안을 논의했다.
군단은 이번 세미나에서 도출된 방안을 토대로 산·학·연 기관과 적극 공조하면서 세미나 개최를 정례화하도록 추진할 방침이다.
실제로 군단 예하 2신속대응사단에서는 신속획득사업 일환으로 오는 12월부터 수소동력 경전술차량과 수소충천차량을 시범운용할 예정이다. 군단 전장이동통제대는 10월 호국훈련에 전장순환통제용 수소드론을 운용할 준비를 하고 있다.
박 군단장은 “미래 수소 모빌리티로의 전환은 선택이 아닌 필수”라고 강조하면서 “수소기술을 통해 기계화부대의 군사작전 효율성과 생존성을 높여 미래전장을 선도할 것으로 기대한다”고 말했다. 이어 “전군에서 가장 많은 기동장비를 운용하는 7군단이 군내 수소 모빌리티 적용에 최적의 테스트베드이자 퍼스트무버로서 역할을 해낼 것”이라고 덧붙였다. 
 - 배지열 기자 -
'''
def sum_body(text):
    result = ''
    # 문장별 인코딩 및 Embedding 추출
    sentences = split_into_sentences(text)
    sentence_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        outputs = model(**inputs)
        sentence_embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy())

    # 문장 중요도 평가 (CLS 토큰의 L2 norm을 사용)
    sentence_scores = [np.linalg.norm(embedding) for embedding in sentence_embeddings]
    top_n = 3  # 추출할 문장 수
    top_sentence_indices = np.argsort(sentence_scores)[-top_n:][::-1]  # 상위 N개 문장 인덱스
    top_sentences = [sentences[idx] for idx in top_sentence_indices]

    
    for sentence in top_sentences:
        result += (sentence + "\n")
    result = result.strip("\n")

    return result

# sum_body(document)

# # 각 문장의 키워드 추출
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
# feature_array = np.array(tfidf_vectorizer.get_feature_names_out())

# for idx in top_sentence_indices:
#     tfidf_sorting = np.argsort(tfidf_matrix[idx].toarray()).flatten()[::-1]
#     top_n_keywords = feature_array[tfidf_sorting][:3]  # 각 문장에서 상위 3개 키워드
#     print(f"문장: {sentences[idx]}")
#     print("키워드:", top_n_keywords)
