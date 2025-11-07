import os
import logging
from typing import List, Tuple
from logging import Logger

from pydantic import BaseModel

# ------------------ URL --------------------
# 다나와
DANAWA = "https://prod.danawa.com/"
# 다나와 URL
DANAWA_URLS  = {
    # 가전
    "TV": "https://prod.danawa.com/list/?cate=10248425",
    "드럼세탁기": "https://prod.danawa.com/list/?cate=102206",
    "건조기": "https://prod.danawa.com/list/?cate=10221615",
    "청소기": "https://prod.danawa.com/list/?cate=10254654",
    # 전자기기
    "노트북": "https://prod.danawa.com/list/?cate=112758",
    "모니터": "https://prod.danawa.com/list/?cate=112757",
    "휴대폰": "https://prod.danawa.com/list/?cate=122515",
    "스마트워치": "https://prod.danawa.com/list/?cate=12215657",
    # 스포츠
    "러닝화": "https://prod.danawa.com/list/?cate=13255574",
    "운동화/스니커즈": "https://prod.danawa.com/list/?cate=13252352",
    "등산화/트래킹화": "https://prod.danawa.com/list/?cate=13227854",
    "축구화": "https://prod.danawa.com/list/?cate=13227989",
    # 가구
    "침대": "https://prod.danawa.com/list/?cate=15241194",
    "매트리스/토퍼": "https://prod.danawa.com/list/?cate=15239795",
    "소파": "https://prod.danawa.com/list/?cate=15236036",
    "학생/사무용의자" : "https://prod.danawa.com/list/?cate=1523647",
    # 식품
    "헬스/다이어트식품" : "https://prod.danawa.com/list/?cate=16254123",
    "홍삼" : "https://prod.danawa.com/list/?cate=16253962",
    "유산균": "https://prod.danawa.com/list/?cate=16242109",
    "비타민/미네랄": "https://prod.danawa.com/list/?cate=1622278",
    # 유아/완구
    "유모차" : "https://prod.danawa.com/list/?cate=16249192",
    "레고/블럭" : "https://prod.danawa.com/list/?cate=16249274",
    "역할놀이/소꿉놀이" : "https://prod.danawa.com/list/?cate=16249298",
    "신생아/영유아완구": "https://prod.danawa.com/list/?cate=16249390",
    # 뷰티
    "기초세트": "https://prod.danawa.com/list/?cate=18255390",
    "스킨/토너": "https://prod.danawa.com/list/?cate=18255391",
    "로션": "https://prod.danawa.com/list/?cate=18255394",
    "크림/수딩젤": "https://prod.danawa.com/list/?cate=18255395",
}

# ------------------ 카테고리 --------------------
# 다나와 카테고리 이름, 파일명 매핑
DANAWA_CATEGORIES: list[Tuple[str, str]] = [
    # 가전
    ("TV", "tv"),
    ("드럼세탁기", "drum_washer"),
    ("건조기", "dryer"),
    ("청소기", "vacuum_cleaner"),
    # 전자기기
    ("노트북", "laptop"),
    ("모니터", "monitor"),
    ("휴대폰", "smartphone"),
    ("스마트워치", "smartwatch"),
    # 스포츠
    ("러닝화", "running_shoes"),
    ("운동화/스니커즈", "sneakers"),
    ("등산화/트래킹화", "hiking_shoes"),
    ("축구화", "soccer_shoes"),
    # 가구
    ("침대", "bed"),
    ("매트리스/토퍼", "mattress_topper"),
    ("소파", "sofa"),
    ("학생/사무용의자", "office_chair"),
    # 식품
    ("헬스/다이어트식품", "health_diet_food"),
    ("홍삼", "red_ginseng"),
    ("유산균", "probiotics"),
    ("비타민/미네랄", "vitamin_mineral"),
    # 유아/완구
    ("유모차", "stroller"),
    ("레고/블럭", "lego_blocks"),
    ("역할놀이/소꿉놀이", "role_play_toys"),
    ("신생아/영유아완구", "infant_toys"),
    # 뷰티
    ("기초세트", "basic_skincare_set"),
    ("스킨/토너", "skin_toner"),
    ("로션", "lotion"),
    ("크림/수딩젤", "cream_soothing_gel"),
]

# ------------------ 데이터 구조 정의 --------------------
class ReviewEntry(BaseModel):
    pid: str
    product: str
    user_id: str
    score: str
    mall: str
    date: str
    comment: str

    def as_csv_row(self) -> List[str]:
        return [self.pid, self.product, self.user_id, self.score, self.mall, self.date, self.comment]

# ------------------ 로그 정의 --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def configure_root_logger() -> None:
    """
    프로젝트 최상위에서 한 번만 호출해서 root logger를 설정합니다.
    (앱 진입점에서 실행하도록 하면 됩니다.)
    """
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # 이미 핸들러가 붙어 있으면 중복 등록을 방지
    if root.handlers:
        return

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

def get_logger(name: str) -> Logger:
    """
    모듈/클래스 별로 로거를 가져올 때 사용하는 헬퍼 함수.
    configure_root_logger()가 이미 호출되어 있어야 함.
    """
    return logging.getLogger(name)