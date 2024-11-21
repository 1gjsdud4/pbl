import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_TOKEN")  # OpenAI API 키 설정

# MongoDB 연결 설정
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["vector_database"]
collection = db["vector_store"]

# CSV 파일 경로 설정
CSV_FILE_PATH = "2022.csv"  # 업로드할 CSV 파일 경로

# 임베딩 생성 함수
def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"  # 지정한 모델 사용
        )
        embedding = response.data[0].embedding
        return embedding
    
    except Exception as e:
        print(f"Error generating embedding for text: {text}\n{e}")
        return None

# CSV 데이터를 업로드하는 함수
def upload_csv_with_openai_embeddings(csv_file_path):
    # 기존 데이터 삭제
    print("Deleting existing documents in the collection...")
    collection.delete_many({})  # 기존 데이터 모두 삭제
    print("All existing documents have been deleted.")

    # CSV 파일 읽기
    data = pd.read_csv(csv_file_path)

    # 데이터베이스에 저장할 문서 목록 생성
    documents = []
    for _, row in data.iterrows():
        # 성취기준 내용을 OpenAI 임베딩으로 벡터화
        embedding = get_embedding(row["성취기준 내용"])
        if embedding is None:  # 에러 발생 시 건너뛰기
            continue

        document = {
            "교과": row["교과"],
            "과목명": row["과목명"],
            "학교급": row["학교급"],
            "성취기준 번호(코드)": row["성취기준 번호(코드)"],
            "성취기준 내용": row["성취기준 내용"],
            "embedding": embedding,
        }
        documents.append(document)

    # MongoDB에 데이터 업로드
    if documents:
        result = collection.insert_many(documents)
        print(f"{len(result.inserted_ids)} documents with embeddings uploaded to MongoDB.")
    else:
        print("No documents were uploaded due to errors.")

if __name__ == "__main__":
    upload_csv_with_openai_embeddings(CSV_FILE_PATH)
