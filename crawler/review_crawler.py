# 표준 라이브러리
import csv
import time
import os
import sys
import random
import re
from multiprocessing import Pool
from typing import Iterable, List, Tuple
from pathlib import Path

# 서드파티 라이브러리
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException,
    NoSuchElementException,
    TimeoutException,
    ElementClickInterceptedException,
)

# 외부 라이브러리
from utils import configure_root_logger, get_logger
from utils import DANAWA, DANAWA_URLS, DANAWA_CATEGORIES
from utils import ReviewEntry

# 디렉토리 경로
BASE_DIR = Path(__file__).resolve().parent.parent # Project Directory  
WORK_DIR = BASE_DIR / "crawler" # 이 파일이 있는 폴더
DATA_DIR = BASE_DIR / "data"
CHROMEDRIVER_PATH = WORK_DIR / ("chromedriver.exe" if os.name == "nt" else "chromedriver")

# ------------------ 로그 설정 --------------------
configure_root_logger()
logger = get_logger("review_crawler")

# ------------------ 크롤러 --------------------
class ReviewCrawler:
    def __init__(self):
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-notifications")
        # options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")  # 최대화 모드로 시작
        options.add_argument("lang=ko_KR")

        service = Service(executable_path=str(CHROMEDRIVER_PATH))
        self.driver = Chrome(service=service, options=options)
        self.wait = WebDriverWait(self.driver, 10)

    # --------- public API -------------
    def crawl(self, cat_name: str, cat_code: str) -> None:
        try:
            url = DANAWA_URLS[cat_name]
            logger.info(f"[{cat_name}] 시작")
            self.driver.get(url)
            # self._select_90_per_page()            
            self._select_sort_method()
            pids = self._collect_product_id()
            entries = self._collect_reviews(cat_name, pids)

            self._save_to_csv(rows=entries, code=cat_code)
            # logger.info(f"[{cat_name}] 완료 ({len(entries):,}개)")

        except Exception as err:
            logger.error(f"[{cat_name}] 실패: {err!r}")
        finally:
            self.driver.quit()

    # ------- 1 Page Item 90개 설정 ----------
    def _select_90_per_page(self) -> None:
        try:
            self.driver.find_element(By.XPATH, "//option[@value='90']").click()
            self.wait.until(
                EC.invisibility_of_element_located(
                    (By.CSS_SELECTOR, ".product_list_cover")
                )
            )
        except Exception as e:
            logger.debug("[페이지당 90개] 설정 실패: %s", e)
    
    def _select_sort_method(self) -> None:
        try:
            self.driver.find_element(By.XPATH, "//li[@data-sort-method='BoardCount']").click()
            self.wait.until(
                EC.invisibility_of_element_located(
                    (By.CSS_SELECTOR, ".product_list_cover")
                )
            )
        except Exception as e:
            logger.debug("[상품평 많은 순] 설정 실패: %s", e)

    def _collect_product_id(self) -> list[str]:
        pids = []
        container_css = "div.main_prodlist.main_prodlist_list"
        list_css = f"{container_css} ul.product_list"

        self.wait.until(
            EC.invisibility_of_element_located(
                (By.CSS_SELECTOR, ".product_list_cover")
            )
        )
        self.wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "li[id^='productInfoDetail_']")
            )
        )
        # Lazy_load 대기
        self._wait_items_stable(list_css)

        items = self.driver.find_elements(
                By.CSS_SELECTOR, "input[id^='productItem_categoryInfo_']"
        )

        for li in items:
            try:
                pid = li.get_attribute("id").split("_")[2]
                pids.append(pid)
            except Exception:
                continue
        
        return pids

    
    # --------- 리뷰 수집 ----------------------
    def _collect_reviews(self, cat_name: str, pids: list[str]) -> list[ReviewEntry]:
        match = re.search(r"cate=(\d+)", DANAWA_URLS[cat_name])
        cate_num = None
        if match:
            cate_num = match.group(1)
        entries: list[ReviewEntry] = []
        total = len(pids)
        step = max(1, total // 10) # 20등분
        start = time.perf_counter()

        for idx, pid in enumerate(pids, 1):
            time.sleep(random.uniform(5, 10))  # polite delay
            review_link = f"https://prod.danawa.com/info/?pcode={pid}&cate={cate_num}&deliveryYN=N&bookmark=cm_opinion&companyReviewYN=Y#bookmark_cm_opinion"
            try:
                self._open_in_new_tab(review_link)

                self.wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "#danawa_container")
                    )
                )
                self.wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "#danawa-prodBlog-companyReview-content-list")
                    )
                )

                # 제품 이름
                product = self.driver.find_element(
                    By.CSS_SELECTOR, ".top_summary .prod_tit .title"
                ).text.strip()

                # review page
                max_page = 4
                for next_page in range(2, max_page, 1):
                    self.wait.until(
                        EC.presence_of_element_located(
                                (By.CSS_SELECTOR, ".rvw_list")
                            )
                    )
                    time.sleep(random.uniform(1, 2))  # polite delay
                    items = self.driver.find_elements(
                        By.CSS_SELECTOR, ".danawa-prodBlog-companyReview-clazz-more"
                    )
                    try:
                        for item in items:
                            score = item.find_element(
                                By.CSS_SELECTOR, ".top_info .star_mask"
                            ).text.strip()
                            mall = item.find_element(
                                By.CSS_SELECTOR, ".top_info .mall img"
                            ).get_attribute("alt")
                            date = item.find_element(
                                By.CSS_SELECTOR, ".top_info .date"
                            ).text.strip()
                            user_id = item.find_element(
                                By.CSS_SELECTOR, ".top_info .name"
                            ).text.strip()
                            comment = item.find_element(
                                By.CSS_SELECTOR, ".rvw_atc .atc_cont .atc"
                            ).text.strip()
                            entries.append(ReviewEntry(pid=pid, product=product, user_id = user_id, score = score,
                                                    mall = mall, date = date, comment = comment))
                    except Exception as e:
                        logger.info("[리뷰 탭 처리 실패] %s (%s)", e, product)

                    try:
                        next_btn = self.wait.until(
                            EC.element_to_be_clickable(
                                (By.XPATH, f'//div[@class="page_nav_area"]//div[contains(@class,"nums_area")]//a[@data-pagenumber="{next_page}"]')
                            )
                        )
                        next_btn.click()
                    except Exception as e:
                        logger.info("페이지 이동 실패 %s (%s)", e, cat_name)
                        break

                if idx % step == 0 or idx == total:
                    pct = idx / total * 100  # 0-100 %
                    elapsed = time.perf_counter() - start
                    logger.info(
                        "[%s] %.1f %% (%d/%d) - %.0f s 경과",
                        cat_name,
                        pct,
                        idx,
                        total,
                        elapsed,
                    )

            except Exception as e:
                logger.debug("[리뷰 창 열기 실패] %s (%s)", e, review_link)

            finally:
                # 탭 정리 후 리스트 페이지로 복귀
                if len(self.driver.window_handles) > 1:
                    self._close_current_tab()

        return entries
    
    def _open_in_new_tab(self, url: str):
        """새 탭으로 열고 포커스 이동"""
        self.driver.execute_script("window.open(arguments[0], '_blank');", url)
        self.driver.switch_to.window(self.driver.window_handles[-1])

    def _close_current_tab(self):
        """현재 탭 닫고 첫 탭(리스트)로 복귀"""
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[0])

    # -------- 행 안정 대기 ---------------------
    def _wait_items_stable(
        self, list_css: str, timeout: int = 10, interval: float = 0.3
    ):
        end = time.time() + timeout
        prev = -1
        while time.time() < end:
            cur = len(
                self.driver.find_elements(
                    By.CSS_SELECTOR, f"{list_css} li[id^='productInfoDetail_']"
                )
            )
            if cur == prev and cur > 0:
                return
            prev = cur
            time.sleep(interval)
        raise TimeoutException("아이템 개수 안정 대기 시간 초과")
    

    # ------- csv 저장 -------------------
    def _save_to_csv(self, rows: List[ReviewEntry], code: str) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, f"{code}_reviews.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Pid", "Product", "User_id", "Score", "Mall", "Date", "Comment"])
            writer.writerows(entry.as_csv_row() for entry in rows)


# -------- 멀티 프로세스 진입 -------------
def _worker(cat: Tuple[str, str]) -> None:
    name, code = cat
    ReviewCrawler().crawl(name, code)


def main():
    processes = 2
    with Pool(processes) as pool:
        pool.map(_worker, DANAWA_CATEGORIES)
    logger.info(" 모든 카테고리 크롤링 완료")


if __name__ == "__main__":
    main()
