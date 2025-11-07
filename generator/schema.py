# LLM 응답 구조 정의
from pydantic import BaseModel, Field

class LGCParam(BaseModel):
    # 프롬프트에 파라미터로 넣을 인자는 여기서 넣는다.
    product_title: str
    category: str
    stars: int = Field(description="20|40|60|80|100 중 하나")
    target_len: int = Field(description="원하는 출력 길이(문자 수)")

class LGCResponse(BaseModel):
    answer: str = Field(description="리뷰 본문(설명/메타 텍스트 없이 한 개만)")

