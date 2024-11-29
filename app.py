from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pytz
import os
from langchain.memory import ConversationBufferMemory
import openai

# 성취기준 출처 https://innovalley.tistory.com/40

#시간 함수
def time_now():

    now_ny = datetime.now(pytz.timezone('Asia/Seoul'))

    # 날짜 및 시간 형식 지정
    formatted_time = now_ny.strftime("%Y-%m-%d %H:%M:%S %Z")
    return formatted_time

load_dotenv()  # .env 파일에서 환경 변수 로드
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_TOKEN')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # 비밀 키 설정

# LLM 설정
llm = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')
chat_llm = ChatOpenAI(temperature=1.0, model='gpt-4o')

# MongoDB 연결 설정
client = MongoClient(os.getenv('MONGO_URI'))
db = client["pbl_database"]
users_collection = db["users"]
chats_collection = db["chats"]
vb = client["vector_database"]
vector_collection =vb["vector_store"]


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

# 성취기준 검색
def search_criteria_by_conversation(conversation_history, top_n=5):
    # 대화 내용을 하나의 텍스트로 결합하고 벡터화
    full_conversation = " ".join(conversation_history)
    query_embedding = get_embedding(full_conversation)

    if query_embedding is None:
        return []

    # MongoDB에서 필요한 필드만 가져오기
    cursor = vector_collection.find({}, {"embedding": 1, "other_fields": 1})

    # 임베딩과 문서 메타데이터를 분리
    embeddings = []
    documents = []

    for doc in cursor:
        # 임베딩을 NumPy 배열로 변환하여 추가
        embeddings.append(np.array(doc["embedding"]))
        documents.append(doc)

    # 유사도 계산
    query_embedding = np.array(query_embedding).reshape(1, -1)
    embeddings_array = np.vstack(embeddings)
    similarities = cosine_similarity(query_embedding, embeddings_array).flatten()

    # 유사도를 문서에 추가
    for idx, doc in enumerate(documents):
        doc["similarity"] = similarities[idx]

    # 상위 N개 결과 반환
    top_results = sorted(documents, key=lambda x: x["similarity"], reverse=True)[:top_n]
    return top_results

# 사용자 생성 함수
def create_user(name, student_id):
    user_data = {
        "_id": student_id,
        "name": name,
        "created_at": time_now()
    }
    try:
        users_collection.insert_one(user_data)
        print(f"User created: {name} ({student_id})")
    except Exception as e:
        print(f"Failed to create user: {e}")

# 사용자 조회 함수
def get_user(student_id):
    user = users_collection.find_one({"_id": student_id})
    return user


# 로그인 페이지
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name")
        student_id = request.form.get("student_id")

        user = get_user(student_id)
        if user:
            if user['name'] == name:
                session['student_id'] = student_id
                session['name'] = name
                return redirect(url_for('chat'))
            else:
                error = "학번과 이름이 일치하지 않습니다."
                return render_template("login.html", error=error)
        else:
            create_user(name, student_id)
            session['student_id'] = student_id
            session['name'] = name
            return redirect(url_for('chat'))
    else:
        return render_template("login.html")





# 채팅 페이지
@app.route("/chat")
def chat():
    if 'student_id' not in session:
        return redirect(url_for('login'))
    student_id = session['student_id']
    name = session['name']
    # 사용자의 대화 목록 가져오기
    chats = list(chats_collection.find({"user_id": student_id}, {"_id": 1, "title": 1}))
    
    if not chats:
        # 대화 세션이 없으면 새로 생성
        default_title = "새로운 대화"
        chat_id = str(ObjectId())
        chat_data = {
            "_id": chat_id,
            "user_id": student_id,
            "title": default_title,
            "messages": [],
            "last_updated": time_now()
        }
        chats_collection.insert_one(chat_data)
        chats.append({"_id": chat_id, "title": default_title})
        current_chat_id = chat_id
    else:
        # 기존 대화가 있으면 가장 최근 대화를 현재 대화로 설정
        current_chat = chats[-1]
        current_chat_id = current_chat["_id"]
        
    return render_template("chat.html", name=name, chats=chats, current_chat_id=current_chat_id)

# 새로운 대화 생성
@app.route("/new_chat", methods=["POST"])
def new_chat():
    if 'student_id' not in session:
        return redirect(url_for('login'))
    student_id = session['student_id']
    # 새로운 대화 세션 생성
    chat_id = str(ObjectId())
    title = "새로운 대화"  # 기본 제목으로 설정
    chat_data = {
        "_id": chat_id,
        "user_id": student_id,
        "title": title,
        "messages": [],
        "last_updated": time_now()
    }
    chats_collection.insert_one(chat_data)
    current_chat_id = chat_id
    return jsonify({"chat_id": current_chat_id, "title": title})

@app.route("/get_first_message", methods=["POST"])
def get_first_message():
    """
    첫 메시지를 반환하는 엔드포인트
    """
    data = request.json  # 클라이언트로부터 JSON 데이터 받기
    chat_id = data.get("chat_id")  # chat_id 추출

    if not chat_id:
        return jsonify({"error": "chat_id가 제공되지 않았습니다."}), 400

    # chat_id를 이용해 대화 내용을 검색하거나 필요한 데이터를 처리
    chat = chats_collection.find_one({"_id": chat_id})
    if not chat:
        return jsonify({"error": "chat_id에 해당하는 대화가 없습니다."}), 404

    # 첫 메시지 생성 (예: 이전 대화 맥락 기반 또는 기본 메시지)
    first_message = "지금 PBL 설계 어디까지 되었나요? 😊"
    
    return jsonify({"message": first_message})

# 특정 대화 불러오기
@app.route("/get_chat/<chat_id>")
def get_chat(chat_id):
    if 'student_id' not in session:
        return redirect(url_for('login'))
    
    # MongoDB에서 대화 기록을 가져옵니다.
    chat = chats_collection.find_one({"_id": chat_id})
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    
    messages = chat["messages"]
    
    # JSON 형태로 대화 기록을 반환
    return jsonify({"messages": messages})

memory = ConversationBufferMemory()

# 메시지 전송 및 응답 처리
@app.route("/api/<chat_id>", methods=["POST"])
def api(chat_id):
    if 'student_id' not in session:
        return redirect(url_for('login'))
    student_id = session['student_id']
    user_message = request.json.get("message")
    question = user_message
    if not chat_id:
        return jsonify({"error": "Chat not found"}), 404
    
    chat = chats_collection.find_one({"_id": chat_id})
    
    # 이전 메시지 불러오기
    messages = chat.get("messages", [])
    # RunnableWithMessageHistory 초기화
    persona_prompt = """
            당신은 "PBL 교수설계 보조 챗봇"입니다. 설계자가 효과적으로 PBL 수업을 설계할 수 있도록 각 단계별로 체계적인 질문을 제공하고 스캐폴딩을 지원하는 역할을 수행합니다.
            * 대화를 자연스럽고 친근하게 시작하며, PBL 교수설계의 각 단계에서 설계자가 질문에 답하며 설계를 구체화할 수 있도록 돕습니다.
            * 질문은 한 개만 한다. 
            * 설계자가 작성한 답변을 토대로 추가 질문을 제공하여 작업을 심화합니다.

        INITIAL_ANSWER:
        {initial_answer}

        CHAT_HISTOY:
        {chat_history}

"""




    # 사용자 메시지 추가
    messages.append({
        "role": "user",
        "content": user_message,
        "timestamp": time_now(),
    })




    chat_prompt_1 = """
        GOAL:
        * 설계자가 PBL 활동을 진행할 교과, 학년, 단원 주제, 학습 목표를 설정하고 점검할 수 있도록 정보를 제공합니다.
        USER:
        {question}
        FEW_SHOT:
        1. 학생의 수준은 어떠한가요?
        2. 이 PBL 수업을 적용하려는 교과와 단원은 무엇인가요?
        3. 대상 학년과 학습자의 수준을 고려하여 PBL 활동의 주요 주제를 정리해 보세요.
        4. 이 활동이 어떤 학습 목표와 연계되어 있는지 명확히 기술해 주세요.
        성취기준:
        * 2022 교육과정 성취기준을 확인할 수 있도록 한다.
        * https://ncic.re.kr/new/mobile.dwn.ogf.inventoryList.do 사이트를 안내한다.

        
        CHAT_HISTOY:
        {chat_history}

        
        OUTPUT:
        """
        

    chat_prompt_2 = """
        GOAL:
        * 학습자가 문제를 해결하는 과정에서 달성해야 할 학습목표를 설정하고, 목표가 구체적이고 측정 가능하도록 안내합니다.
        USER:
        {question}
        FEW_SHOT:
        1. 학생들이 문제를 해결하면서 어떤 지식, 기술, 또는 태도를 습득해야 하나요?
        2. 학습 목표가 구체적이고 측정 가능한지 점검해 보세요.
        3. 이 목표가 PBL 활동을 통해 자연스럽게 달성될 수 있는지 확인하세요.
        EXAMPLE:
        1. 지식: 물리적 에너지원의 변환 과정을 설명할 수 있다.
        기술: 실험 데이터를 분석하고 결론을 도출할 수 있다.
        태도: 지속 가능한 에너지 사용의 중요성을 인식한다.
        2. 지식: 역사적 사건 간의 인과관계를 분석할 수 있다.
        기술: 문헌을 분석하고, 다양한 관점을 비교하며 의견을 논리적으로 제시할 수 있다.
        태도: 다문화적 관점을 수용하며 열린 자세를 유지한다.

        
        CHAT_HISTOY:
        {chat_history}
   
    OUTPUT:
      """



    chat_prompt_3 = """
        GOAL:
        * 설계자가 학습자가 문제를 해결하는 데 필요한 주요 개념, 절차, 원리를 정리할 수 있도록 돕습니다.
        USER:
        {question}
        FEW_SHOT:
        1. 설계하려는 문제에서 학생들이 반드시 이해해야 할 개념은 무엇인가요?
        2. 문제에서 다룰 절차는 무엇인가요?
        3. 학생들이 문제를 해결하면서 배우게 될 절차는 무엇인가요?
        4. 적용해야 할 주요 원리는 무엇인지 명확히 기술해 주세요.
        EXAMPLE:
        1. 개념: 빛의 굴절
        절차: 빛의 굴절 실험 설계 → 데이터 수집 → 결과 분석
        원리: 빛의 매질 변화에 따른 굴절 각도의 변화
        2. 개념: 경제 순환
        절차: 생산자, 소비자, 정부 간 상호작용 분석 → 데이터 시각화 → 결론 도출
        원리: 한정된 자원의 배분과 최적화 원칙

        
        CHAT_HISTOY:
        {chat_history}

        OUTPUT:
    """
        
        

    chat_prompt_4 = """
        GOAL:
        *설계자가 학생들의 실제 삶의 맥락과 밀접하게 연결된 문제를 구체적으로 만들 수 있도록 합니다. 
        *학습자가 몰입할 수 있는 실제 삶에 있을 것 같은 시나리오를 완성할 수 있도록 안내합니다. 
        *설계자는 학습자가 자기주도적 학습을 통해 문제를 탐구하고 해결안을 제시할 수 있는 환경을 조성해야 합니다. 
        *문제는 복잡하고 비구조화되어 있어 학습자 스스로 탐구와 논의를 통해 해결 방안을 찾을 수 있도록 해야 합니다. 
        USER:
        {question}
        FEW_SHOT:
        1. 문제 상황을 구성하려면 현실적이고 실제적인 맥락을 상상해 보세요.
        2. 문제에서 다루어야 할 내용을 실제로 활용하고 있는 사람들은 누구일까요?
        3. 학생들이 문제를 해결하는 과정에서 겪을 어려움은 무엇인가요?
        4. 문제 해결 시 학생들이 답해야 할 핵심 질문은 무엇인가요?
        5. 문제를 소개하는 시나리오를 작성해 보세요. 학생들이 문제를 이해하고 몰입할 수 있도록, 현실적인 배경이나 흥미로운 스토리를 포함하면 좋습니다.
        6. 시나리오에서 학생들이 맡게 될 역할은 무엇인가요?
        7. 시나리오가 학습자의 흥미를 유발하고 동기를 부여할 수 있는지 점검해 보세요.
        8. 문제 상황에서 학습자는 어떤 맥락에서 어떤 역할을 하고 있나요? 
        EXAMPLE:
        1. 문제 상황:
        "도시 중심부에 위치한 오래된 공원이 시설 노후화로 인해 방문자가 급감하고 있습니다. 도시 정부는 공원을 다시 활성화하기 위해 혁신적인 아이디어를 찾고 있습니다. 학생들은 도시 설계 전문가로서 공원을 재구성하는 계획을 설계해야 합니다."
        주요 질문:
        공원의 이용률을 높이기 위한 창의적인 디자인은 무엇인가?
        공원 시설 개선에 필요한 자원을 어떻게 확보할 것인가?
        재구성된 공원이 지역 주민들에게 제공할 수 있는 이점은 무엇인가?

        2. 문제 상황:
        "지역의 중소기업들이 디지털 전환 과정에서 어려움을 겪고 있습니다. 특히, 효율적인 온라인 마케팅 전략을 찾는 데 큰 장애가 되고 있습니다. 학생들은 중소기업 컨설턴트로서 디지털 전환을 지원할 수 있는 실질적인 전략을 설계해야 합니다."
        주요 질문:
        중소기업이 저비용으로 활용할 수 있는 효과적인 온라인 마케팅 도구는 무엇인가?
        제안된 전략이 중소기업의 매출에 어떤 영향을 미칠 것인가?
        기업들이 디지털 전환의 주요 장애물을 극복할 수 있도록 어떤 지원이 필요한가?

        3. 문제 상황:
        "지역의 고등학생들이 스마트폰 중독 문제로 학업 성취도가 감소하고 있습니다. 지역 교육청은 이 문제를 해결하기 위해 혁신적인 프로그램을 필요로 합니다. 학생들은 교육 전문가로서 스마트폰 중독을 완화하고 학업 성취도를 높일 수 있는 프로그램을 설계해야 합니다."
        주요 질문:
        스마트폰 사용을 줄이기 위한 창의적인 활동은 무엇인가?
        프로그램의 효과를 학업 성취도와 연결지을 수 있는 방법은 무엇인가?
        제안된 프로그램이 학생들의 일상생활에 자연스럽게 적용될 수 있는가?

        
        CHAT_HISTOY:
        {chat_history}

        OUTPUT:

    """
        


    chat_prompt_5 = """
            GOAL:
            * 설계자가 학생들이 제출해야 할 최종 결과물을 명확히 설계할 수 있도록 지원합니다.
            USER:
            {question}
            FEW_SHOT:
            1. 학생들이 문제를 해결한 후 제출해야 할 최종 과제는 무엇인가요? (예: 보고서, 발표, 영상)
            2. 이 과제가 학습 목표를 달성했는지 평가할 수 있는 요소를 포함하고 있나요?
            EXAMPLE:
            1. 과제 형태: 팀 보고서
            내용: 에너지 절약 방안을 설계하고, 이를 적용한 시뮬레이션 결과를 보고서로 작성한다.
            평가 요소:
            - 보고서 구조의 논리성
            - 제안한 방안의 창의성 및 타당성
            - 데이터 분석과 시각화의 적절성
            2. 과제 형태: 발표 자료
            내용: 새로운 대중교통 시스템의 설계 방안을 발표하고, 예상되는 환경적 효과를 설명한다.
            평가 요소:
            - 발표 자료의 시각적 효과
            - 제안 내용의 실현 가능성
            - 팀원 간 협업 수준

            
        CHAT_HISTOY:
        {chat_history}

            OUTPUT:
    """
        




    chat_prompt_6 = """
            GOAL:
            * 설계자가 학습자들에게 문제를 제시하기 위한 시나리오 구현 방법을 설계하도록 돕습니다.
            * 시나리오를 학습자에게 효과적으로 전달하기 위한 다양한 구현 방안을 제안하고, 선택된 방식의 장단점을 점검할 수 있는 가이드와 질문을 제공합니다.
            USER:
            {question}
            FEW_SHOT:
            1. 시나리오를 학습자들에게 전달하는 가장 적합한 방법은 무엇인가요? (예: 동영상, 이메일, 포스터, 인터뷰 자료 등)
            2. 선택한 방법이 시나리오의 맥락과 학습 목표에 부합하는지 점검해 보세요.
            학습자들이 시나리오에 몰입할 수 있도록 추가적인 자료(사진, 그래프, 뉴스 기사 등)를 활용하는 방안을 고려해 보세요.
            3. 선택한 구현 방법의 장점과 단점은 무엇인가요?
            4. 시나리오 구현 방법이 학습자의 참여를 유도할 수 있는지 점검해 보세요.
            EXAMPLE:
            1. 시나리오 구현 방법: 동영상
            설명:
            "지역 해안 생태계가 심각하게 훼손된 상황을 담은 3분 길이의 다큐멘터리 스타일 동영상을 제작하여 학습자들에게 전달한다. 동영상에는 사진 자료, 인터뷰 클립, 통계 그래프 등이 포함된다."
            장점:
            시각적 자료를 통해 학습자들의 몰입도를 높인다.
            복잡한 문제를 쉽게 이해할 수 있도록 돕는다.
            단점:
            제작 시간이 오래 걸릴 수 있다.
            기술적 문제가 발생할 경우, 전달에 지장이 생길 수 있다.

            2. 시나리오 구현 방법: 가상 이메일
            설명:
            "학습자들이 환경 보호 단체의 팀원이라는 설정으로, 단체 대표로부터 시급한 문제를 다룬 이메일을 받는 방식이다. 이메일에는 문제 상황 설명, 주요 데이터, 역할 배정, 기대되는 결과에 대한 간단한 안내가 포함된다."
            장점:
            간단하고 시간 효율적이다.
            학습자들에게 실제 업무 환경과 비슷한 몰입감을 제공한다.
            단점:
            시각적 자료의 부재로 학습자들의 흥미를 덜 자극할 수 있다.

            3. 시나리오 구현 방법: 가상 인터뷰 자료
            설명:
            "학생들이 가상의 전문가(농부, 정책 분석가, 과학자)와 인터뷰를 진행한 것처럼 설정된 대화형 자료를 제공한다. 인터뷰 내용에는 문제 상황과 주요 이슈에 대한 배경 설명이 담긴다."
            장점:
            대화형 자료를 통해 학습자들이 문제를 다양한 관점에서 접근할 수 있다.
            문제에 대한 실제적이고 인간적인 연결 고리를 제공한다.
            단점:
            인터뷰 자료를 작성하는 데 시간이 소요될 수 있다.

            
        CHAT_HISTOY:
        {chat_history}

            OUTPUT:

    """
        

 
    chat_prompt_7 = """
            GOAL:
            * 학생들이 문제를 해결하기 위해 필요한 정보와 자원을 체계적으로 정리할 수 있도록 돕습니다.
            * 정보 자원의 종류(문서, 데이터, 동영상, 도구 등)와 이를 학습자들에게 효과적으로 제공할 방법을 제안합니다.
            * 학습자가 문제 해결에 필요한 정보를 찾기 위해 시간을 낭비하지 않도록 사전에 적절한 자료를 제공하도록 지원합니다.
            USER:
            {question}
            FEW_SHOT:
            1. 학생들이 문제를 해결하기 위해 반드시 필요한 정보와 자료는 무엇인가요? (학습이 아닌 문제 해결을 위한 자료로 한정하여 생각해 보세요.)
            2. 제공하려는 정보 자원이 학습자가 직접 찾는 데 시간을 허비하지 않도록 준비되었나요?
            3. 문제 해결 과정에서 필요할 수 있는 다양한 자원(데이터, 사례 연구, 시뮬레이션 도구 등)이 충분히 포함되었는지 점검해 보세요.
            4. 정보 자원이 학습 목표와 문제 해결 과정에 적합한지 확인해 보세요.
            자료를 학습자가 쉽게 접근할 수 있는 형태(링크, 문서, 동영상 등)로 준비했나요?
            5. 학습자가 제공된 자원을 효과적으로 활용할 수 있도록 추가적인 가이드를 포함했나요?
            EXAMPLE:
            1. 정보 자원:
            - 학술 자료: 기후 변화 보고서, 에너지 절약 관련 연구 논문.
            - 동영상: 친환경 에너지 기술 관련 다큐멘터리.
            - 데이터: 지역 에너지 소비 데이터와 CO2 배출량 통계.
            2. 정보 자원:
            - 학습 자료: 쓰레기 재활용 사례 연구, 관련 환경 정책 요약서.
            - 시뮬레이션 도구: 쓰레기 처리 효율을 계산하는 소프트웨어.
            - 참고 사이트: UN 기후 변화 보고 웹사이트.

            
        CHAT_HISTOY:
        {chat_history}

            OUTPUT:
    """       



    chat_prompt_8 = """
            GOAL:
            * 설계자가 학생들이 문제를 해결하는 과정을 단계별로 설계할 수 있도록 지원합니다. 각 단계에서 학생들이 수행할 활동과 필요한 지원을 구체화하도록 돕습니다.
            USER:
            {question}
            OUTPUT:
            1. 학생들이 문제 해결을 시작할 때 어떤 정보를 제공해야 하나요?
            2. 각 단계에서 학생들이 수행할 주요 활동은 무엇인가요?
            3. 문제 해결 과정에서 팀 구성과 역할 분담을 어떻게 설계할 것인지 고민해 보세요.
            EXAMPLE:
            1. 수행 단계:
            - 단계 1: 문제 분석 및 목표 정의 (팀 회의로 문제를 분석하고 목표를 설정).
            - 단계 2: 정보 검색 및 자료 수집 (팀별로 할당된 자료를 수집하고 정리).
            - 단계 3: 아이디어 도출 및 실행 계획 수립 (브레인스토밍으로 해결 방안 도출).
            - 단계 4: 결과물 작성 및 발표 준비 (보고서 작성 및 발표 자료 제작).
            
            2. 팀 구성:
            - 팀원 1: 자료 검색 및 분석 담당.
            - 팀원 2: 아이디어 발표 및 실행 계획 작성.
            - 팀원 3: 최종 결과물 제작 및 발표 준비.


        CHAT_HISTOY:
        {chat_history}

            OUTPUT:
    """

    

    chat_prompt_9 = """
            GOAL:
            * 설계자가 문제 해결 활동에 필요한 소요시간을 추정할 수 있도록 지원합니다. 활동별 예상 시간을 작성하도록 안내하며, 학습 목표에 적합한지 점검할 수 있도록 돕습니다.
            USER:
            {question}
            FEW_SHOT:
            1. 각 문제 해결 단계에 예상 소요시간을 설정해 보세요.
            2. 학생들의 집중력과 학습 목표에 적합한 시간을 배정했는지 점검해 보세요.
            EXAMPLE:
            1. 소요시간:
            - 문제 분석 및 목표 정의: 30분.
            - 정보 검색 및 자료 수집: 1시간 30분.
            - 아이디어 도출 및 실행 계획 수립: 1시간.
            - 결과물 작성 및 발표 준비: 2시간.
            2. 소요시간:
            - 단계 1: 팀 회의로 목표 설정 (20분).
            - 단계 2: 학술 자료 검색 (1시간).
            - 단계 3: 시뮬레이션 실행 및 결과 분석 (2시간).
            - 단계 4: 발표 준비 및 시연 (1시간 30분).

            
        CHAT_HISTOY:
        {chat_history}

            OUTPUT:
    """
        



    chat_prompt_10 = """
            GOAL:
            * 설계자가 학생들의 학습 결과를 평가할 계획을 설계하도록 돕습니다. 
            * 학습 목표와 평가 기준의 일치 여부를 점검하고, 평가가 다양한 관점(교수자 평가, 팀 간 평가, 팀원 평가)을 반영하도록 안내합니다.
            USER:
            {question}
            FEW_SHOT:
            1. 평가 기준이 학습 목표와 충분히 연계되어 있는지 점검해 보세요.
            2. 평가 계획에 교수자 평가, 팀 간 평가, 팀원 평가 요소를 모두 포함했는지 확인해 보세요.
            3. 평가 기준이 학습 결과의 다양성을 반영할 수 있도록 창의성, 협업 능력, 논리적 사고 등을 고려해 설계하세요.
            4. 평가 방법이 공정성과 명확성을 유지하도록 평가 항목별 세부 기준을 작성해 보세요.
            EXAMPLE:
            1. 평가 기준: 지속 가능한 에너지 사용 방안 설계 프로젝트
            학습 목표: 지속 가능한 에너지 사용 방안을 제시할 수 있다.
            교수자 평가:
            창의적이고 혁신적인 아이디어 (30점).
            데이터 분석의 정확성 (30점).
            해결 방안의 실행 가능성과 현실성 (20점).
            발표 자료의 구성 및 완성도 (20점).
            팀 간 평가:
            다른 팀의 발표에서 학습한 점을 피드백으로 작성 (10점).
            상대 팀 해결 방안의 설득력과 참신성 평가 (10점).
            팀원 평가:
            팀 내 역할 분담의 적절성 평가.
            각 팀원이 협업 과정에서 기여한 정도를 서로 평가 (5점).
            2. 평가 기준: 지역사회 문제 해결 프로젝트
            학습 목표: 지역사회의 문제를 분석하고 실행 가능한 해결책을 제시한다.
            교수자 평가:
            문제 분석의 깊이와 논리적 전개 (25점).
            실행 방안의 실현 가능성 (30점).
            발표의 논리적 전개와 설득력 (25점).
            시각 자료의 효과적 사용 (20점).
            팀 간 평가:
            팀원들이 다른 팀 발표에서 도출한 개선점과 배운 점을 토론 (10점).
            다른 팀의 해결 방안에 대한 질문과 피드백 (10점).
            팀원 평가:
            개인별 역할 수행 평가 (10점).
            팀워크 기여도 평가 (5점).
            
            
        CHAT_HISTOY:
        {chat_history}
            
            OUTPUT:
    """


    chat_prompt_11 = """
        USER:
        {question}

        OUTPUT:
    """ 


    # 단계 분석 템플릿
    state_prompt =  """
        GOAL:
        * 당신은 "PBL 교수설계 보조자"입니다.
        * 설계자가 제공한 질문과 정보를 바탕으로 PBL 교수설계 단계 중 하나를 추론하세요.
        * 방금까지 단계의 목표가 도달하면 다음 단계로 추론합니다. 
        * 단계는 다음 중 하나로 답변합니다:
        - '교과 및 단원'
        - '학습목표'
        - '문제에서 다룰 내용'
        - '문제'
        - '최종과제'
        - '시나리오 구현방법'
        - '필요한 정보자원'
        - '문제해결 수행방법'
        - '문제해결 소요시간'
        - '평가계획’

        FORMAT:
        * 예: {{"type": "교과 및 단원"}}, {{"type": "시나리오 구현방법"}}
        USER INPUT:
        {question}
        OUTPUT:

    """


    state_prompt = ChatPromptTemplate.from_template(
        state_prompt
    )

    state_chain = state_prompt | llm | StrOutputParser()


    
    result = state_chain.invoke({ 'question': question})
    
    if result:
        json_result = json.loads(result)
    print(result)



    if json_result['type'] == '교과 및 단원':
        chat_prompt =  chat_prompt_1
    elif json_result['type'] == '학습목표':
        chat_prompt = chat_prompt_2

    elif json_result['type'] == '문제에서 다룰 내용':
        chat_prompt = chat_prompt_3

    elif json_result['type'] == '문제':
        chat_prompt = chat_prompt_4

    elif json_result['type'] == '최종과제':
        chat_prompt = chat_prompt_5

    elif json_result['type'] == '시나리오 구현방법':
        chat_prompt =  chat_prompt_6

    elif json_result['type'] == '필요한 정보자원':
        chat_prompt =  chat_prompt_7

    elif json_result['type'] == '문제해결 수행방법':
        chat_prompt =chat_prompt_8

    elif json_result['type'] == '문제해결 소요시간':
        chat_prompt = chat_prompt_9

    elif json_result['type'] == '문제해결 소요시간':
        chat_prompt = chat_prompt_10

    else:
        chat_prompt = chat_prompt_11


    chat_prompt = ChatPromptTemplate.from_template(
            chat_prompt
        )

    chat_chain = (
    RunnablePassthrough()  # 사용자 입력 그대로 전달
    | chat_prompt  # 프롬프트 적용
    | chat_llm   # OpenAI 모델 실행
    | StrOutputParser()  
)   # 최근 N개의 메시지만 포함
    N = 30  # 메시지 개수 제한
    limited_messages = messages[-N:]  # 최근 N개의 메시지 선택
    at_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in limited_messages])
    second_result = chat_chain.invoke({
    "question": question,
    "chat_history": at_history_text
})
    chat_prompt =  ChatPromptTemplate.from_template(
            persona_prompt
            
        )

    chat_chain_2 = (
    RunnablePassthrough()  # 사용자 입력 그대로 전달
    | chat_prompt  # 프롬프트 적용
    | chat_llm   # OpenAI 모델 실행
    | StrOutputParser() 

    ) 
    
    
    final_result = chat_chain_2.invoke({
        "initial_answer": second_result,
        "chat_history": at_history_text
    })
        
    print("Final Result:", final_result)

    messages.append({
        "role": "assistant",
        "content": final_result,
        "timestamp": time_now(),
    })
        

    # 대화 세션 업데이트 
    chats_collection.update_one(
        {"_id": chat_id},
        {
            "$set": {
                "messages": messages,
                "last_updated": time_now()
            }
        }
    )

    # **여기에서 첫 번째 응답이면 제목 생성**
    try:
        new_title = None
        if len(messages) < 3:  # 사용자의 첫 메시지와 챗봇의 첫 응답이 추가된 상태
            # 제목 생성 프롬프트
            title_prompt = """
        GOAL:
        * 다음 대화의 내용을 기반으로 대화의 제목을 10자 이내로 생성합니다.
        * 제목은 핵심 내용을 담고 있어야 합니다.
        * 단어만 반환하고, 불필요한 문장이나 설명은 하지 마세요.

        CHAT HISTORY:
        {chat_history}

        OUTPUT:
        """
            title_prompt = ChatPromptTemplate.from_template(title_prompt)

            title_chain = title_prompt | llm | StrOutputParser()
            at_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            generated_title = title_chain.invoke({'chat_history': at_history_text})
            # 대화 세션의 제목 업데이트
            chats_collection.update_one(
                {"_id": chat_id},
                {
                    "$set": {"title": generated_title.strip()}
                }
            )
            new_title = generated_title
            print(new_title)
    except Exception as e:
        print(f"Error generating title: {e}")

    # JSON 응답 반환
    return jsonify({"message": final_result, "title": new_title})

if __name__ == '__main__':
    app.run(host='0.0.0.0')


