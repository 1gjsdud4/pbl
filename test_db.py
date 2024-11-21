from pymongo import MongoClient
from datetime import datetime, timezone
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
# MongoDB 연결 설정

load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))

# 데이터베이스 및 컬렉션 설정
db = client["pbl_database"]  # 데이터베이스 이름
users_collection = db["users"]  # 사용자 정보 컬렉션
chats_collection = db["chats"]  # 대화 세션 컬렉션

# 사용자 생성 함수
def create_user(name, student_id):
    """
    사용자 정보를 MongoDB에 저장합니다.
    :param name: 사용자 이름
    :param student_id: 학번
    """
    user_data = {
        "_id": student_id,  # 학번을 고유 ID로 사용
        "name": name,
        "created_at": datetime.now(timezone.utc)  # UTC 시간 저장
    }
    try:
        users_collection.insert_one(user_data)
        print(f"User created: {name} ({student_id})")
    except Exception as e:
        print(f"Failed to create user: {e}")

# 사용자 조회 함수
def get_user(student_id):
    """
    학번으로 사용자를 조회합니다.
    :param student_id: 학번
    """
    user = users_collection.find_one({"_id": student_id})
    if user:
        print(f"User found: {user['name']} ({user['_id']})")
        return user
    else:
        print(f"No user found with student_id: {student_id}")
        return None

# 대화 세션 생성 함수
def create_chat_session(student_id, title):
    """
    새로운 대화 세션을 생성합니다.
    :param student_id: 학번
    :param title: 대화 제목
    """
    chat_data = {
        "_id": str(ObjectId()),  # 고유 대화 ID
        "user_id": student_id,
        "title": title,
        "messages": [],
        "last_updated": datetime.now(timezone.utc)  # UTC 시간 저장
    }
    chats_collection.insert_one(chat_data)
    print(f"Chat session created for student_id {student_id} with title '{title}'.")

# 메시지 추가 함수
def add_message_to_chat(chat_id, role, content):
    """
    대화 세션에 메시지를 추가합니다.
    :param chat_id: 대화 ID
    :param role: 메시지 역할 ('user' 또는 'assistant')
    :param content: 메시지 내용
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)  # UTC 시간 저장
    }
    chats_collection.update_one(
        {"_id": chat_id},
        {
            "$push": {"messages": message},
            "$set": {"last_updated": datetime.now(timezone.utc)}  # 업데이트 시간 수정
        }
    )
    print(f"Message added to chat {chat_id}: {role} - {content}")

# 사용자 대화 목록 조회 함수
def get_user_chats(student_id):
    """
    사용자의 모든 대화 세션을 조회합니다.
    :param student_id: 학번
    """
    chats = list(chats_collection.find({"user_id": student_id}, {"_id": 1, "title": 1, "last_updated": 1}))
    if chats:
        print(f"Chat sessions for student_id {student_id}:")
        for chat in chats:
            print(f"- Chat ID: {chat['_id']}, Title: {chat['title']}, Last Updated: {chat['last_updated']}")
    else:
        print(f"No chat sessions found for student_id {student_id}.")
    return chats

# 대화 세부정보 조회 함수
def get_chat_details(chat_id):
    """
    특정 대화 세션의 세부 내용을 조회합니다.
    :param chat_id: 대화 ID
    """
    chat = chats_collection.find_one({"_id": chat_id})
    if chat:
        print(f"Chat details for chat ID {chat_id}:")
        for message in chat["messages"]:
            print(f"[{message['timestamp']}] {message['role']}: {message['content']}")
        return chat
    else:
        print(f"No chat found with chat ID: {chat_id}")
        return None

# 사용자 삭제 함수
def delete_user(student_id):
    """
    사용자를 삭제합니다.
    :param student_id: 학번
    """
    result = users_collection.delete_one({"_id": student_id})
    if result.deleted_count > 0:
        print(f"User with student_id {student_id} deleted.")
    else:
        print(f"No user found with student_id {student_id}.")

# 모든 대화 세션 삭제 함수
def delete_all_chats():
    """
    모든 대화 세션을 삭제합니다.
    """
    result = chats_collection.delete_many({})
    print(f"Deleted {result.deleted_count} chat sessions.")

# 단독 실행 시 테스트
if __name__ == "__main__":
    print("==== MongoDB Test ====")
    
    # 테스트 사용자
    student_id = "20230001"
    name = "John Doe"

    print("\n1. 사용자 생성")
    create_user(name, student_id)

    print("\n2. 사용자 조회")
    get_user(student_id)

    print("\n3. 대화 세션 생성")
    create_chat_session(student_id, "First Conversation")

    print("\n4. 대화 목록 조회")
    chats = get_user_chats(student_id)

    print("\n5. 메시지 추가")
    if chats:
        chat_id = chats[0]["_id"]  # 첫 번째 대화 세션 ID
        add_message_to_chat(chat_id, "user", "안녕하세요!")
        add_message_to_chat(chat_id, "assistant", "안녕하세요, 무엇을 도와드릴까요?")

    print("\n6. 대화 세부내용 조회")
    if chats:
        get_chat_details(chats[0]["_id"])

    print("\n7. 사용자 삭제")
    delete_user(student_id)

    print("\n8. 모든 대화 세션 삭제")
    delete_all_chats()


# 단독 실행 시 테스트
if __name__ == "__main__":
    print("==== MongoDB Test ====")
    
    # 테스트 사용자
    student_id = "20230001"
    name = "John Doe"

    print("\n1. 사용자 생성")
    create_user(name, student_id)

    print("\n2. 사용자 조회")
    get_user(student_id)

    print("\n3. 대화 세션 생성")
    create_chat_session(student_id, "First Conversation")

    print("\n4. 대화 목록 조회")
    chats = get_user_chats(student_id)

    print("\n5. 메시지 추가")
    if chats:
        chat_id = chats[0]["_id"]  # 첫 번째 대화 세션 ID
        add_message_to_chat(chat_id, "user", "안녕하세요!")
        add_message_to_chat(chat_id, "assistant", "안녕하세요, 무엇을 도와드릴까요?")

    print("\n6. 대화 세부내용 조회")
    if chats:
        get_chat_details(chats[0]["_id"])
