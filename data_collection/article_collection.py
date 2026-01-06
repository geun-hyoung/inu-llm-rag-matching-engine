"""
EBSCO 논문 데이터 수집기
MariaDB의 v_emp1_3 테이블에서 논문 정보를 가져와
EBSCO에서 논문 메타데이터를 수집합니다.
"""

import mariadb
import pandas as pd
from playwright.sync_api import sync_playwright
import time
import json
from pathlib import Path
import sys

# 상위 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from config.database import get_db_connection, close_db_connection, get_article_data, COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM, COL_ARTICLE_PUBLSH_DT
from config.settings import ARTICLE_DATA_FILE

# DB 연결 및 데이터 조회
conn = None
try:
    conn = get_db_connection()
    
    # 논문 데이터 조회 (2015년 이상)
    print("\n📚 논문 데이터 조회 중...")
    df_emp = get_article_data(conn, min_year=2015)
    
    print(f"2015년 이상 논문 필터링 후: {len(df_emp)}개")
    print(f"중복 제거 후: {len(df_emp)}개")
    print(f"게재일자 순 정렬 완료 (최신순)")
    
    # 확인
    print("\n총 (EMP_NO, THSS_NM) 세트 수:", len(df_emp))
    print("\n[미리보기 TOP 10 - 최신순]")
    print(df_emp[[COL_ARTICLE_EMP_NO, COL_ARTICLE_THSS_NM, COL_ARTICLE_PUBLSH_DT]].head(10))

except mariadb.Error as e:
    print("MariaDB 연결 실패!")
    print("오류 코드:", e.errno)
    print("오류 메시지:", e.msg)
    sys.exit(1)

EBSCO_URL = "https://research.ebsco.com/c/4zvbuh/search"

# df_emp 에 THSS_NM과 EMP_NO를 함께 사용
# (EMP_NO, THSS_NM) 쌍으로 저장 (THSS_NM 중복 가능하지만 EMP_NO는 다를 수 있음)
queries_list = []
for _, row in df_emp.iterrows():
    queries_list.append({
        "EMP_NO": str(row[COL_ARTICLE_EMP_NO]).strip(),
        "THSS_NM": str(row[COL_ARTICLE_THSS_NM]).strip()
    })

# THSS_NM 기준으로 중복 제거 (같은 제목이 여러 EMP_NO에 있을 수 있으므로 첫 번째 것만 사용)
seen_titles = set()
unique_queries = []
for q_dict in queries_list:
    if q_dict["THSS_NM"] not in seen_titles and q_dict["THSS_NM"]:
        seen_titles.add(q_dict["THSS_NM"])
        unique_queries.append(q_dict)

queries = unique_queries
total_queries = len(queries)
print(f"\n총 {total_queries}개의 논문 제목을 검색합니다.")

# 기존 결과 파일이 있으면 로드 (중간 저장용)
output_file = Path(ARTICLE_DATA_FILE)
output_file.parent.mkdir(parents=True, exist_ok=True)
existing_results = []
processed_titles = set()

try:
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
        processed_titles = set([str(item.get('THSS_NM', '')) for item in existing_results if item.get('THSS_NM')])
        print(f"기존 파일에서 {len(existing_results)}개의 결과를 불러왔습니다.")
        print(f"이미 처리된 논문: {len(processed_titles)}개")
        # 기존 파일에 EMP_NO가 없으면 추가 (호환성을 위해)
        if existing_results and 'EMP_NO' not in existing_results[0].keys():
            print("  ⚠️ 경고: 기존 파일에 EMP_NO 컬럼이 없습니다. 새 데이터부터 EMP_NO가 포함됩니다.")
except FileNotFoundError:
    print("기존 결과 파일이 없습니다. 새로 시작합니다.")
except Exception as e:
    print(f"기존 파일 로드 중 오류 (무시하고 계속 진행): {e}")

results = existing_results.copy() if existing_results else []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # 헤드리스 모드로 실행 (창 숨김)
    context = browser.new_context()
    page = context.new_page()

    page.goto(EBSCO_URL, wait_until="domcontentloaded")
    # input("검색 가능한 상태면 Enter...")  # 헤드리스 모드에서는 불필요

    # 중간 저장 주기 (5개마다 저장)
    save_interval = 5
    start_idx = len(results)  # 이미 처리된 개수
    
    try:
        for i, q_dict in enumerate(queries, 1):
            q = q_dict["THSS_NM"]  # 검색어는 THSS_NM
            emp_no = q_dict["EMP_NO"]  # EMP_NO 저장용
            
            # 이미 처리된 논문은 건너뛰기
            if str(q) in processed_titles:
                print(f"[{i}/{total_queries}] 이미 처리됨 - 건너뜀: {q[:50]}...")
                continue
            
            print(f"[{i}/{total_queries}] Searching: {q}")

            try:
                # 현재 URL 확인 - 홈화면이면 검색 페이지로 이동
                current_url = page.url
                if "search" not in current_url:
                    print(f"  → 홈화면 감지, 검색 페이지로 이동 중...")
                    page.goto(EBSCO_URL, wait_until="domcontentloaded")
                    time.sleep(1.5)
                
                # 검색 입력 필드가 존재하는지 확인
                try:
                    search_input = page.locator("input#search-input")
                    if search_input.count() == 0:
                        print(f"  → 검색 입력 필드 없음, 검색 페이지로 이동 중...")
                        page.goto(EBSCO_URL, wait_until="domcontentloaded")
                        time.sleep(1.5)
                except:
                    print(f"  → 검색 입력 필드 확인 실패, 검색 페이지로 이동 중...")
                    page.goto(EBSCO_URL, wait_until="domcontentloaded")
                    time.sleep(1.5)

                # 검색 입력 필드 클리어 후 검색
                try:
                    page.click("input#search-input")
                    time.sleep(0.5)
                    # 전체 선택 후 삭제
                    page.keyboard.press("Control+A")
                    time.sleep(0.3)
                    page.keyboard.press("Backspace")
                    time.sleep(0.5)
                except:
                    pass
                
                # 검색어 입력
                page.fill("input#search-input", q)
                time.sleep(0.5)
                page.press("input#search-input", "Enter")

                # 결과 로딩 대기 (더 충분한 대기 시간)
                page.wait_for_load_state("networkidle")
                time.sleep(3.0)  # SPA 렌더 안정화 (헤드리스 모드에서는 더 긴 대기 필요)
                
                # 검색 결과 요소가 나타날 때까지 명시적으로 대기 시도 (최대 5초)
                try:
                    # 검색 결과 또는 "결과 없음" 메시지가 나타날 때까지 대기
                    page.wait_for_selector('h3[data-auto="result-item-title"], p:has-text("철자를 확인하거나"), text=/검색 결과.*건/', timeout=5000)
                except:
                    # 요소가 나타나지 않아도 계속 진행 (이미 networkidle로 충분히 대기했으므로)
                    pass
                
                time.sleep(1.0)  # 추가 안정화 대기

                # 홈화면으로 리다이렉트되었는지 다시 확인
                current_url_after = page.url
                if "search" not in current_url_after:
                    print(f"  → 검색 후 홈화면으로 이동됨 - 검색 실패")
                    results.append({
                        "EMP_NO": emp_no,
                        "THSS_NM": q,
                        "has_result": 0
                    })
                    processed_titles.add(str(q))
                    continue

                has_result = 0  # 기본값: 없음

                # ✅ 검색 결과 있음 확인: 여러 방법으로 시도 (재시도 포함)
                # 방법 1: data-auto 속성 사용 (가장 안정적)
                result_cnt_1 = page.locator('h3[data-auto="result-item-title"]').count()
                result_cnt_link = page.locator('a[data-auto="result-item-title__link"]').count()
                
                # 방법 2: mark 태그 확인 (검색 결과가 있을 때 하이라이트되는 mark 태그)
                # h3 내부의 mark 태그만 확인하여 더 정확하게 판단
                result_cnt_mark_in_h3 = page.locator('h3[data-auto="result-item-title"] mark').count()
                result_cnt_mark_all = page.locator('mark').count()
                
                # 방법 3: 클래스명 패턴 사용 (부분 일치 - 백업용)
                result_cnt_2 = page.locator('div[class*="result-item-header__title"]').count()
                result_cnt_3 = page.locator('div[class*="result-item-header"]').count()
                
                # 방법 4: 검색 결과 개수 텍스트 확인 ("검색 결과: X건")
                result_count_text = page.locator('text=/검색 결과.*건/').count()
                
                # 첫 번째 시도에서 아무것도 찾지 못하면 추가 대기 후 재시도
                if (result_cnt_1 == 0 and result_cnt_link == 0 and result_cnt_mark_in_h3 == 0 and 
                    result_cnt_mark_all < 2 and result_count_text == 0):
                    time.sleep(2.0)  # 추가 대기 후 재시도
                    result_cnt_1 = page.locator('h3[data-auto="result-item-title"]').count()
                    result_cnt_link = page.locator('a[data-auto="result-item-title__link"]').count()
                    result_cnt_mark_in_h3 = page.locator('h3[data-auto="result-item-title"] mark').count()
                    result_cnt_mark_all = page.locator('mark').count()
                    result_cnt_2 = page.locator('div[class*="result-item-header__title"]').count()
                    result_cnt_3 = page.locator('div[class*="result-item-header"]').count()
                    result_count_text = page.locator('text=/검색 결과.*건/').count()
                
                # 최종 결과 개수 판단
                # h3 요소가 있거나, 링크가 있으면 검색 결과가 있다고 판단
                result_cnt = max(result_cnt_1, result_cnt_link, result_cnt_2, result_cnt_3)
                
                # h3 내부의 mark 태그가 1개 이상 있으면 확실히 검색 결과가 있다고 판단
                # (제공된 HTML 구조: h3 안에 여러 mark 태그가 있음)
                if result_cnt_mark_in_h3 >= 1:
                    result_cnt = max(result_cnt, 1)
                # 전체 mark 태그가 2개 이상 있으면 검색 결과가 있을 가능성이 높음
                elif result_cnt_mark_all >= 2:
                    result_cnt = max(result_cnt, 1)
                
                # 검색 결과 개수 텍스트가 있으면 결과가 있다고 판단
                if result_count_text > 0:
                    result_cnt = max(result_cnt, 1)
                
                # ✅ 검색 결과 없음 확인: 텍스트 기반으로 확인
                no_result_cnt = page.locator('text=/철자를 확인하거나.*검색하십시오/').count()
                if no_result_cnt == 0:
                    no_result_cnt = page.locator('p:has-text("철자를 확인하거나")').count()
                
                # 홈화면인지 확인 (검색 입력 필드가 없거나 검색 페이지가 아닌 경우)
                is_home_page = "search" not in page.url or page.locator("input#search-input").count() == 0

                # 디버깅 정보 출력
                if result_cnt == 0 and no_result_cnt == 0:
                    print(f"  → 디버깅: h3-title={result_cnt_1}, link={result_cnt_link}, header-title={result_cnt_2}, header={result_cnt_3}, mark-in-h3={result_cnt_mark_in_h3}, mark-all={result_cnt_mark_all}, result-count-text={result_count_text}, no-result={no_result_cnt}, is-home={is_home_page}")

                if is_home_page:
                    has_result = 0
                    print("  → 홈화면으로 이동함 (검색 실패) (0)")
                    results.append({
                        "EMP_NO": emp_no,
                        "THSS_NM": q,
                        "has_result": 0
                    })
                    processed_titles.add(str(q))  # 처리 완료 표시
                elif no_result_cnt > 0:
                    has_result = 0
                    print("  → 검색 결과 없음 (0)")
                    # 검색 결과 없음도 JSON에 저장 (재실행 시 다시 검색하지 않도록)
                    results.append({
                        "EMP_NO": emp_no,
                        "THSS_NM": q,
                        "has_result": 0
                    })
                    processed_titles.add(str(q))  # 처리 완료 표시
                elif result_cnt > 0 or result_cnt_mark_in_h3 >= 1 or result_cnt_mark_all >= 2 or result_count_text > 0:
                    has_result = 1
                    print(f"  → 검색 결과 있음 (1) - h3={result_cnt_1}개, link={result_cnt_link}개, mark-in-h3={result_cnt_mark_in_h3}개, mark-all={result_cnt_mark_all}개, 결과텍스트={result_count_text}개")
                    
                    # ✅ 검색 결과가 있으면 첫 번째 결과의 상세 페이지로 이동하여 메타데이터 추출
                    paper_metadata = {"EMP_NO": emp_no, "THSS_NM": q, "has_result": 1}
                    
                    try:
                        # 첫 번째 결과 링크 찾기
                        first_link = page.locator('a[data-auto="result-item-title__link"]').first
                        if first_link.count() > 0:
                            # 새 탭에서 열기 (또는 현재 페이지에서 이동)
                            href = first_link.get_attribute("href")
                            if href:
                                # 상대 경로를 절대 경로로 변환
                                if href.startswith("/"):
                                    detail_url = "https://research.ebsco.com" + href
                                else:
                                    detail_url = href
                                
                                # 상세 페이지로 이동
                                page.goto(detail_url, wait_until="domcontentloaded")
                                time.sleep(1.5)  # 페이지 로딩 대기
                                
                                # 메타데이터 추출
                                metadata_div = page.locator('div[data-auto="record-html-metadata"] article')
                                if metadata_div.count() > 0:
                                    # JavaScript로 메타데이터 추출 (더 안정적)
                                    metadata_dict = page.evaluate("""
                                        () => {
                                            const article = document.querySelector('div[data-auto="record-html-metadata"] article');
                                            if (!article) return {};
                                            
                                            const result = {};
                                            const h3Elements = article.querySelectorAll('h3');
                                            
                                            h3Elements.forEach(h3 => {
                                                const key = h3.textContent.trim();
                                                // h3 다음의 첫 번째 ul 찾기
                                                let nextSibling = h3.nextElementSibling;
                                                while (nextSibling && nextSibling.tagName !== 'UL') {
                                                    nextSibling = nextSibling.nextElementSibling;
                                                }
                                                
                                                if (nextSibling && nextSibling.tagName === 'UL') {
                                                    const liElements = nextSibling.querySelectorAll('li');
                                                    const values = Array.from(liElements).map(li => li.textContent.trim()).filter(v => v);
                                                    if (values.length > 0) {
                                                        result[key] = values.length === 1 ? values[0] : values;
                                                    }
                                                }
                                            });
                                            
                                            return result;
                                        }
                                    """)
                                    
                                    # 추출한 메타데이터를 paper_metadata에 추가
                                    paper_metadata.update(metadata_dict)
                                    
                                    extracted_fields = [k for k in paper_metadata.keys() if k not in ['THSS_NM', 'has_result', 'metadata_error', 'EMP_NO']]
                                    print(f"  → 메타데이터 추출 완료: {len(extracted_fields)}개 필드 ({', '.join(extracted_fields[:3])}{'...' if len(extracted_fields) > 3 else ''})")
                                else:
                                    print("  → 메타데이터 요소를 찾을 수 없음")
                                    paper_metadata["metadata_error"] = "메타데이터 요소 없음"
                                
                                # 검색 페이지로 돌아가기
                                page.goto(EBSCO_URL, wait_until="domcontentloaded")
                                time.sleep(0.5)
                            else:
                                print("  → 링크 href를 찾을 수 없음")
                                paper_metadata["metadata_error"] = "링크 없음"
                        else:
                            print("  → 결과 링크를 찾을 수 없음")
                            paper_metadata["metadata_error"] = "링크 요소 없음"
                            
                    except Exception as e:
                        print(f"  → 메타데이터 추출 중 오류: {str(e)}")
                        paper_metadata["metadata_error"] = f"오류: {str(e)}"
                        # 오류가 발생해도 검색 페이지로 돌아가기 시도
                        try:
                            if "search" not in page.url:
                                page.goto(EBSCO_URL, wait_until="domcontentloaded")
                                time.sleep(0.5)
                        except:
                            pass
                    
                    results.append(paper_metadata)
                    processed_titles.add(str(q))  # 처리 완료 표시
                    
                else:
                    has_result = 0
                    print("  → 결과 판별 실패(로딩/구조 문제) (0) - JSON에 저장하지 않음")
                    # 결과 판별 실패인 경우 results에 추가하지 않음 (기록만 하고 저장은 안 함)
                    processed_titles.add(str(q))  # 처리 완료 표시만 (중복 방지)

                # 주기적으로 중간 저장 (오류 발생 시 데이터 손실 방지)
                # 결과 판별 실패는 저장하지 않으므로 results에는 항상 유효한 데이터만 있음
                current_processed = len(results) - start_idx
                if current_processed > 0 and current_processed % save_interval == 0:
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"  💾 중간 저장 완료 ({len(results)}개 결과 저장됨)")
                    except Exception as e:
                        print(f"  ⚠️ 중간 저장 실패 (계속 진행): {e}")

                # 다음 검색 준비 (홈화면이 아닐 때만 클리어)
                current_url_before_clear = page.url
                if "search" in current_url_before_clear and page.locator("input#search-input").count() > 0:
                    try:
                        clear_btn = page.locator('button[aria-label="Clear"]')
                        if clear_btn.count() > 0:
                            clear_btn.click()
                            time.sleep(0.5)
                        else:
                            # Clear 버튼이 없으면 수동으로 클리어
                            page.click("input#search-input")
                            time.sleep(0.3)
                            page.keyboard.press("Control+A")
                            time.sleep(0.2)
                            page.keyboard.press("Backspace")
                            time.sleep(0.3)
                    except Exception as e:
                        # 클리어 실패 시 수동으로 클리어
                        try:
                            page.click("input#search-input")
                            time.sleep(0.3)
                            page.keyboard.press("Control+A")
                            time.sleep(0.2)
                            page.keyboard.press("Backspace")
                            time.sleep(0.3)
                        except:
                            pass

                time.sleep(2.0)  # 검색 간격 (너무 빠르면 서버가 차단할 수 있음, 2초면 충분)
                
            except Exception as e:
                # 개별 논문 처리 중 오류 발생 (다음 논문으로 계속 진행)
                print(f"  ❌ 오류 발생 (다음으로 진행): {str(e)}")
                error_result = {
                    "EMP_NO": emp_no,
                    "THSS_NM": q,
                    "has_result": 0,
                    "metadata_error": f"처리 중 오류: {str(e)}"
                }
                results.append(error_result)
                processed_titles.add(str(q))  # 처리 완료 표시
                
                # 주기적으로 저장
                current_processed = len(results) - start_idx
                if current_processed > 0 and current_processed % save_interval == 0:
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"  💾 중간 저장 완료 ({len(results)}개 결과 저장됨)")
                    except Exception as save_error:
                        print(f"  ⚠️ 중간 저장 실패: {save_error}")
                
                continue  # 다음 논문으로 계속

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        raise  # KeyboardInterrupt는 다시 발생시켜서 정상 종료 처리
        
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류 발생: {str(e)}")
        print("지금까지 수집한 데이터를 저장합니다...")
        
    finally:
        # 브라우저 종료
        try:
            browser.close()
        except:
            pass
        
        # 최종 결과 저장 (오류 발생해도 저장)
        # 결과 판별 실패는 results에 포함되지 않으므로 저장할 데이터만 있음
        try:
            if results:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n✅ 최종 결과를 '{output_file}' 파일로 저장했습니다. (총 {len(results)}개)")
                print("\n=== 검색 결과 및 메타데이터 (일부) ===")
                # JSON 데이터의 처음 10개만 출력
                for i, item in enumerate(results[:10], 1):
                    print(f"{i}. {item.get('THSS_NM', 'N/A')[:50]}... (has_result: {item.get('has_result', 0)})")
                if len(results) > 10:
                    print(f"... 외 {len(results) - 10}개")
            else:
                print("\n⚠️ 저장할 결과가 없습니다.")
        except Exception as e:
            print(f"\n❌ 최종 저장 중 오류 발생: {e}")
            print("수동으로 results 변수에서 데이터를 확인하세요.")
        finally:
            # DB 연결 종료
            close_db_connection(conn)

