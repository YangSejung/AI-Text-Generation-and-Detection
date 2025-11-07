import os, sys
import asyncio
import random
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
# LLM의 응답은 대개 일반 문자열 형태이지만, JsonOutputParser를 사용하면 이 문자열을 프로그래밍 언어에서 쉽게 조작할 수 있는 JSON 객체(예: Python 딕셔너리)로 변환할 수 있습니다.
from langchain_core.exceptions import OutputParserException
from prompt import LGCPrompt
from schema import LGCResponse
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# == 프로젝트 설정 ==
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import settings

from rag_retrieve import retrieve_similar_comments, make_hwc_block

AVAILABLE_MODELS = ["gpt-5-mini", "gpt-5-nano", "gpt-4o-mini"]

def get_random_model():
    return random.choices(
        AVAILABLE_MODELS,
        weights=[0.3, 0.5, 0.2],
    )[0]

async def generate_review(rag_result: dict) -> dict:
    parser = JsonOutputParser(pydantic_object=LGCResponse)

    system_prompt = SystemMessagePromptTemplate.from_template(LGCPrompt.SYSTEM_PROMPT)

    # 프롬프트 구성. 최종 프롬프트 = 시스템 -> 사용자 입력 순으로 구성.
    prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            # user_prompt에는 네가 human으로 추가적으로 주고 싶은 지시 (ex "지금 생성해")
            # HumanMessagePromptTemplate.from_template("{user_prompt}"),
        ]
    )
    # {format_instructions}: parser 예약
    # {history}, {user_prompt}: prompt 예약
    # 고정 변수 미리 주입. 프롬프트에서 {}로 있는 곳에 넣는 곳
    prompt = prompt.partial(
        style_instruction=LGCPrompt.stars_to_style(int(rag_result["stars"])),
        format_instructions=parser.get_format_instructions(), # json으로 응답
    )

    model_name = get_random_model()
    # 모델 생성
    llm = ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        temperature=1,
        max_tokens=2048,
        streaming=False,
    )

    chain = prompt | llm | parser

    # user_prompt 변수는 위 ChatPromptTemplate에 바인딩됩니다. 직접 실행
    try : 
        response = await chain.ainvoke(
            {
                # HumanMessagePromptTemplate 쪽
                # "user_prompt": "위 조건대로 쇼핑몰 제품 리뷰 한 개 생성해.",

                # SystemMessagePromptTemplate 쪽에서 요구하는 변수들
                "retrieved_hwc_block": rag_result["retrieved_hwc_block"],
                "product_title": rag_result["product_title"],
                "category": rag_result["category"],
                "stars": rag_result["stars"],
                "target_len": rag_result["target_len"],
            }
        )
        print("Response:", response)

    except OutputParserException as e:
        print("------ OutputParserException ------")
        raise

    return {
        "model": model_name,
        "result": response,
    }

async def run_once(product_title: str, category: str, stars: int, target_len: int):
    # 1) Pinecone에서 유사 댓글 검색
    docs = retrieve_similar_comments(product_title, category, stars, k=8)
    # 2) 프롬프트에 넣을 블록 생성
    hwc_block = make_hwc_block(docs, limit=6, max_chars=350)

    # 3) generate_review에 넣을 rag_result 구성
    rag_result = {
        "retrieved_hwc_block": hwc_block,
        "product_title": product_title,
        "category": category,
        "stars": int(stars),
        "target_len": int(target_len),
    }

    out = await generate_review(rag_result)
    return out

# example
# if __name__ == "__main__":
#     result = asyncio.run(
#         run_once(
#             product_title="종근당건강 황제침향단 60환",
#             category="홍삼",          # 업서트 시 메타데이터에 저장한 Category 값 그대로
#             stars=20,
#             target_len=20,
#         )
#     )
#     print(f"app result : {result}")
