"""
知网文献采集器 — 完整实现
========================================
搜索阶段: 参考 https://github.com/xxxxchaos/cnki-mcp-server search.py
  - CSS 选择器 a.fz14 / td.author / td.source / td.date 提取表格行
  - 获取 title/url/authors/source/year 基础字段

详情阶段: 参考 cnki-mcp-server detail.py get_paper_detail
  - 逐篇打开论文详情页 URL, 提取 17 个字段
  - CSS 选择器: .wx-tit h1 / h3.author span a / #ChDivSummary / p.keywords a 等

采集流程: CLC 分类号检索 → 表格提取 URL 列表 → 逐篇打开详情页补全
"""
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from config import RAW_DIR, CRAWLER_CONFIG, CATEGORY_MAPPING_FILE


# ===== 采集字段规范 (17 字段, 参考 cnki-mcp-server detail.py) =====
@dataclass
class Paper:
    """单篇文献 — 对齐 cnki-mcp-server get_paper_detail 的 17 字段"""
    title: str = ""
    authors: str = ""          # " ; " 分隔
    orgs: str = ""             # " ; " 分隔
    source: str = ""           # 期刊来源
    year: str = ""
    volume: str = ""           # 卷
    issue: str = ""            # 期
    pages: str = ""            # 页码
    keywords: str = ""        # " ; " 分隔
    abstract: str = ""
    doi: str = ""
    cited_count: str = ""
    download_count: str = ""
    fund: str = ""             # 基金
    classification: str = ""   # 分类号
    clc_code: str = ""         # 本次检索用的三级类目
    url: str = ""
    uid: str = ""

    def __post_init__(self):
        if not self.uid and self.title:
            self.uid = hashlib.md5(
                (self.title + "|" + self.source + "|" + self.year).encode()
            ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


# ===== 浏览器封装 =====
class CnkiBrowser:
    """Selenium Chrome 封装, 注入反检测"""

    # kns8s 是新版入口, 校园网 IP 直接识别机构身份, 不会跳验证码
    # kns8 旧版入口会被拦截到 verify/home 验证码页, 千万不要用
    ADVANCED_SEARCH_URL = (
        "https://kns.cnki.net/kns8s/AdvSearch?classid=WD0FTY92"
    )

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None

    def launch(self):
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        self.driver = webdriver.Chrome(options=opts)
        # 注入 webdriver 覆盖
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            },
        )
        self.wait = WebDriverWait(self.driver, 15)
        logger.info("Chrome 浏览器已启动 (headless={})", self.headless)

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("浏览器已关闭")

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, *args):
        self.close()


# ===== 采集器主类 =====
class CnkiScraper:
    """知网文献采集器 — 按 CLC 三级类目检索"""

    # 高级检索页 XPath (2026 版 kns8s 页面实测)
    XPATH_FIELD_DROPDOWN = (
        '//*[@id="gradetxt"]//div[contains(@class,"sort-default")]'
    )
    XPATH_FIELD_CLC = '//*[@id="gradetxt"]//li[@data-val="CLC"]'
    XPATH_SEARCH_INPUT = (
        '//*[@id="gradetxt"]//dd[1]//input[@type="text"]'
    )
    XPATH_SEARCH_BTN = (
        '//input[contains(@class,"btn-search")]'
    )

    # 结果页表格 CSS 选择器 (参考 cnki-mcp-server search.py)
    CSS_RESULT_ROWS = (
        '#gridTable table.result-table-list tbody tr'
    )
    CSS_ROW_TITLE = 'a.fz14'
    CSS_ROW_AUTHOR = 'td.author a'
    CSS_ROW_SOURCE = 'td.source a'
    CSS_ROW_DATE = 'td.date'
    CSS_ROW_CITED = 'td.quote a'
    CSS_ROW_DOWNLOAD = 'td.download a.downloadCnt'

    # 详情页 CSS 选择器 (参考 cnki-mcp-server detail.py)
    CSS_D_TITLE = '.wx-tit h1'
    CSS_D_AUTHORS = 'h3.author span a'
    CSS_D_ORGS = 'h3.orgn span a'
    CSS_D_ABSTRACT = '#ChDivSummary'
    CSS_D_KEYWORDS = 'p.keywords a'
    CSS_D_SOURCE = 'div.top-tip a[href*="navi.cnki.net"]'
    CSS_D_PUBINFO_SPANS = 'div.top-tip span'
    CSS_D_DOI_LIS = 'li.top-space'
    CSS_D_CITED = 'span#refs a'
    CSS_D_DOWNLOAD = 'span#DownLoadParts a'
    CSS_D_FUND = 'p.funds span'

    # 翻页
    XPATH_RESULT_COUNT = '//span[@class="pagerTitleCell"]'
    XPATH_NEXT_PAGE = '//a[contains(@class,"page-next") or contains(text(),"下一页")]'

    def __init__(self, browser: CnkiBrowser = None):
        self.browser = browser or CnkiBrowser(headless=CRAWLER_CONFIG["headless"])
        self.seen_uids: set = set()
        self._load_seen()

    # -------- 断点续采 --------
    def _load_seen(self):
        """扫描已有 JSONL, 加载已采集的 uid"""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for f in RAW_DIR.glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self.seen_uids.add(json.loads(line)["uid"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("已加载 {} 条历史记录 (去重集合)", len(self.seen_uids))

    # -------- 检索流程 --------
    def goto_search_page(self):
        """打开高级检索页, 等待 gradetxt 加载"""
        d = self.browser.driver
        d.get(CnkiBrowser.ADVANCED_SEARCH_URL)
        # 等待 gradetxt 容器出现
        WebDriverWait(d, 20).until(
            EC.presence_of_element_located((By.ID, "gradetxt"))
        )
        logger.debug("高级检索页已加载")

    def set_field_to_clc(self):
        """把第一个检索字段设为 CLC 分类号"""
        from selenium.webdriver.common.keys import Keys

        d = self.browser.driver
        wait = self.browser.wait
        # 点击字段下拉按钮 (当前显示"主题"的那个)
        dropdown = wait.until(
            EC.element_to_be_clickable((By.XPATH, self.XPATH_FIELD_DROPDOWN))
        )
        dropdown.click()
        time.sleep(0.5)
        # 点击 data-val="CLC" 的选项
        clc_opt = wait.until(
            EC.element_to_be_clickable((By.XPATH, self.XPATH_FIELD_CLC))
        )
        clc_opt.click()
        # 点完 CLC 后, 下拉框不会自动收起, 遮挡输入框 → 按 ESC 收起
        d.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(0.5)
        logger.debug("字段已切换为 CLC 分类号")

    def input_clc_and_search(self, clc_code: str):
        """输入 CLC 编码并点击检索"""
        d = self.browser.driver
        wait = self.browser.wait
        inp = wait.until(
            EC.element_to_be_clickable((By.XPATH, self.XPATH_SEARCH_INPUT))
        )
        inp.clear()
        inp.send_keys(clc_code)
        time.sleep(0.5)
        # 点击检索按钮
        btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, self.XPATH_SEARCH_BTN))
        )
        btn.click()
        # 等待 gridTable 结果区出现
        WebDriverWait(d, 20).until(
            EC.presence_of_element_located((By.ID, "gridTable"))
        )
        time.sleep(2)
        logger.debug("检索已提交: {}", clc_code)

    def switch_to_detail_mode(self):
        """kns8s 新版默认就是列表模式, 没有详情模式可切 — no-op"""
        logger.debug("默认已是列表模式, 跳过详情切换")

    def set_perpage_50(self):
        """kns8s 新版没有每页条数下拉 — no-op"""
        pass

    def get_total_count(self) -> int:
        """解析总结果数"""
        d = self.browser.driver
        try:
            txt = d.find_element(By.XPATH, self.XPATH_RESULT_COUNT).text
            # "共找到 6,700 条结果"
            num = int("".join(c for c in txt if c.isdigit()))
            return num
        except NoSuchElementException:
            return 0

    # -------- 表格模式单页提取 (CSS 选择器, 参考 search.py) --------
    def _extract_table_page(self, clc_code: str) -> List[Paper]:
        """从搜索结果表格提取基础字段: title/url/authors/source/year"""
        d = self.browser.driver
        rows = d.find_elements(By.CSS_SELECTOR, self.CSS_RESULT_ROWS)
        papers = []
        for tr in rows:
            # 标题 + URL
            try:
                title_el = tr.find_element(By.CSS_SELECTOR, self.CSS_ROW_TITLE)
                title = title_el.text.strip()
                url = title_el.get_attribute("href") or ""
            except NoSuchElementException:
                continue  # 没标题 = 表头或无效行

            paper = Paper(title=title, url=url, clc_code=clc_code)

            # 作者
            author_els = tr.find_elements(By.CSS_SELECTOR, self.CSS_ROW_AUTHOR)
            paper.authors = " ; ".join(
                a.text.strip() for a in author_els if a.text.strip()
            )

            # 来源
            try:
                paper.source = tr.find_element(
                    By.CSS_SELECTOR, self.CSS_ROW_SOURCE
                ).text.strip()
            except NoSuchElementException:
                pass

            # 时间
            try:
                paper.year = tr.find_element(
                    By.CSS_SELECTOR, self.CSS_ROW_DATE
                ).text.strip()
            except NoSuchElementException:
                pass

            papers.append(paper)
        return papers

    # -------- 详情页提取 (参考 detail.py get_paper_detail) --------
    def _fetch_paper_detail(self, paper: Paper) -> Paper:
        """
        打开论文详情页 URL, 提取 17 字段, 补全 paper 对象
        参考 cnki-mcp-server detail.py 的 CSS 选择器
        """
        d = self.browser.driver
        url = paper.url
        if not url:
            return paper

        try:
            # 先访问 CNKI 首页建立会话 (避免直接访问详情页触发验证码)
            d.get("https://www.cnki.net/")
            time.sleep(random.uniform(1, 2))
            # 访问详情页
            d.get(url)
            time.sleep(random.uniform(1.5, 2.5))
        except Exception as e:
            logger.warning("详情页加载失败: {}", e)
            return paper

        # 标题
        try:
            paper.title = d.find_element(
                By.CSS_SELECTOR, self.CSS_D_TITLE
            ).text.strip() or paper.title
        except NoSuchElementException:
            pass

        # 作者
        author_els = d.find_elements(By.CSS_SELECTOR, self.CSS_D_AUTHORS)
        if author_els:
            paper.authors = " ; ".join(
                a.text.strip() for a in author_els if a.text.strip()
            )

        # 机构/单位
        org_els = d.find_elements(By.CSS_SELECTOR, self.CSS_D_ORGS)
        if org_els:
            paper.orgs = " ; ".join(
                o.text.strip() for o in org_els if o.text.strip()
            )

        # 摘要
        try:
            paper.abstract = d.find_element(
                By.CSS_SELECTOR, self.CSS_D_ABSTRACT
            ).text.strip()
        except NoSuchElementException:
            pass

        # 关键词
        kw_els = d.find_elements(By.CSS_SELECTOR, self.CSS_D_KEYWORDS)
        if kw_els:
            kws = [k.text.strip().rstrip(";；") for k in kw_els]
            paper.keywords = " ; ".join(k for k in kws if k)

        # 来源
        try:
            paper.source = d.find_element(
                By.CSS_SELECTOR, self.CSS_D_SOURCE
            ).text.strip().rstrip(" .") or paper.source
        except NoSuchElementException:
            pass

        # 年/卷/期/页 — 遍历 top-tip 中的 span 查找出版信息
        spans = d.find_elements(By.CSS_SELECTOR, self.CSS_D_PUBINFO_SPANS)
        for span in spans:
            txt = span.text.strip()
            # 匹配 "2026, 25(04): 464-468" 这样的出版信息行
            # 跳过不含逗号的, 或纯期刊名的行
            if "," in txt and any(c.isdigit() for c in txt) and "." not in txt.split(",")[0]:
                parts = txt.split(",")
                year_str = parts[0].strip()
                # 只取 4 位数字作为年份
                import re
                year_match = re.search(r'\d{4}', year_str)
                if year_match:
                    paper.year = year_match.group()
                if len(parts) > 1:
                    rest = parts[1].strip()
                    if "(" in rest and ")" in rest:
                        paper.volume = rest.split("(")[0].strip()
                        paper.issue = rest.split("(")[1].split(")")[0].strip()
                    if ":" in rest:
                        pages_str = rest.split(":")[-1].strip()
                        # 去掉尾巴的无关文本
                        for sep in [" 查看该刊", " 查看"]:
                            if sep in pages_str:
                                pages_str = pages_str.split(sep)[0].strip()
                        paper.pages = pages_str
                break

        # DOI
        lis = d.find_elements(By.CSS_SELECTOR, self.CSS_D_DOI_LIS)
        for li in lis:
            txt = li.text.strip()
            if "DOI" in txt:
                doi_val = txt.replace("DOI", "").replace("：", "").replace(":", "").strip()
                if doi_val:
                    paper.doi = doi_val
                break

        # 被引次数
        try:
            paper.cited_count = d.find_element(
                By.CSS_SELECTOR, self.CSS_D_CITED
            ).text.strip()
        except NoSuchElementException:
            pass

        # 下载次数
        try:
            paper.download_count = d.find_element(
                By.CSS_SELECTOR, self.CSS_D_DOWNLOAD
            ).text.strip()
        except NoSuchElementException:
            pass

        # 基金
        try:
            paper.fund = d.find_element(
                By.CSS_SELECTOR, self.CSS_D_FUND
            ).text.strip()
        except NoSuchElementException:
            pass

        # 分类号
        try:
            # 用 XPath 找包含"分类号"文本的 li 下的 p
            cls_li = d.find_element(
                By.XPATH, '//li[contains(text(),"分类号")]//p'
            )
            paper.classification = cls_li.text.strip()
        except NoSuchElementException:
            pass

        # 年份兜底清洗: 从任意文本中提取 4 位年份
        import re
        if paper.year and not paper.year.isdigit():
            m = re.search(r'(19|20)\d{2}', paper.year)
            if m:
                paper.year = m.group()
        # pages 清洗: 去掉尾巴的无关文本
        if paper.pages:
            for sep in [" 查看该刊", " 查看", " 收藏"]:
                if sep in paper.pages:
                    paper.pages = paper.pages.split(sep)[0].strip()

        # 重新计算 uid (因为 title/source/year 可能更新了)
        if paper.title:
            paper.uid = hashlib.md5(
                (paper.title + "|" + paper.source + "|" + paper.year).encode()
            ).hexdigest()[:12]

        return paper

    # -------- 翻页 --------
    def next_page(self) -> bool:
        """点击下一页, 返回是否成功"""
        d = self.browser.driver
        try:
            next_btn = d.find_element(By.XPATH, self.XPATH_NEXT_PAGE)
            # 检查是否可点击 (有些情况下"下一页"是禁用的)
            cls = next_btn.get_attribute("class") or ""
            if "disabled" in cls or "is-disabled" in cls:
                logger.debug("下一页按钮已禁用")
                return False
            next_btn.click()
            time.sleep(random.uniform(3, 6))
            return True
        except NoSuchElementException:
            return False

    # -------- 单类目采集 --------
    def collect_category(
        self, clc_code: str, max_papers: int = 50,
        fetch_detail: bool = True,
    ) -> List[Paper]:
        """
        按 CLC 编码采集, 最多 max_papers 条
        1. 表格提取基础字段 (title/authors/source/year/url)
        2. 逐篇打开详情页补全 17 字段 (abstract/keywords/orgs/doi/...)
        """
        d = self.browser.driver

        # 1. 打开高级检索 + 设字段为 CLC + 输入编码 + 检索
        self.goto_search_page()
        self.set_field_to_clc()
        self.input_clc_and_search(clc_code)

        total = self.get_total_count()
        if total == 0:
            logger.warning("[{}] 无检索结果, 跳过", clc_code)
            return []
        target = min(total, max_papers)
        logger.info("[{}] 共 {} 条, 目标采集 {} 条 (detail={})",
                    clc_code, total, target, fetch_detail)

        # 2. 翻页采集基础字段
        collected: List[Paper] = []
        pages = (target // 20) + 3  # 默认每页 20 条, 多翻一页兜底

        for page in range(1, pages + 1):
            if len(collected) >= target:
                break

            batch = self._extract_table_page(clc_code)
            new = 0
            for p in batch:
                if p.uid in self.seen_uids:
                    continue
                collected.append(p)
                self.seen_uids.add(p.uid)
                new += 1
                if len(collected) >= target:
                    break

            logger.info("[{}] page {}: 提取 {}, 新增 {}, 累计 {}",
                        clc_code, page, len(batch), new, len(collected))

            if len(collected) >= target:
                break
            if not self.next_page():
                break

            # 随机间隔防风控
            time.sleep(random.uniform(*CRAWLER_CONFIG["request_interval"]))

        # 3. 逐篇打开详情页, 补全 17 字段
        if fetch_detail and collected:
            logger.info("[{}] 开始逐篇提取详情 ({} 篇)...", clc_code, len(collected))
            for i, p in enumerate(tqdm(
                collected, desc=f"[{clc_code}] 详情", unit="篇"
            )):
                try:
                    self._fetch_paper_detail(p)
                except Exception as e:
                    logger.warning("[{}] 第 {} 篇详情提取失败: {}",
                                   clc_code, i + 1, e)
                # 随机间隔防风控
                time.sleep(random.uniform(2, 5))

        return collected

    def _extract_year_from_page(self, _title: str) -> str:
        """详情模式的年份嵌在 p[1] 里, 但 XPath 不固定 — 表格模式才独立列
        这里不单独处理年份, 在合并阶段再从 URL / 详情页单独取
        """
        return ""

    # -------- 保存 --------
    def save_papers(self, papers: List[Paper], path: Path = None):
        path = path or (RAW_DIR / "cnki_papers.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

    def save_csv(self, path: Path = None):
        """把所有 JSONL 合并存一个 CSV"""
        path = path or (RAW_DIR / "cnki_papers.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        all_rows = []
        for f in sorted(RAW_DIR.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_rows.append(json.loads(line))
        if not all_rows:
            logger.warning("没有数据可导出 CSV")
            return
        keys = list(all_rows[0].keys())
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        logger.info("已导出 CSV: {} ({} 行)", path, len(all_rows))


# ===== 入口: 遍历所有三级类目 =====
def run_full_crawl(
    max_per_category: int = 50,
    categories: List[str] = None,
    resume: bool = True,
    headless: bool = False,
    save_per_category: bool = True,
    fetch_detail: bool = True,
):
    """
    遍历 category_mapping.json 中所有三级类目, 逐个采集

    Args:
        max_per_category: 每个类目最多采集多少条 (建议 30~100, 看论文总量)
        categories: 指定类目列表, None 则从 category_mapping.json 加载全部
        resume: 是否断点续采 (跳过已有 uid)
        headless: 是否无头浏览器 (调试时 False 可看画面)
        save_per_category: 是否每个类目单独存一个 jsonl (便于分片处理)

    Returns:
        dict: 统计信息
    """
    # 加载类目
    if categories is None:
        mapping = json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))
        categories = sorted(mapping.keys())

    logger.info("=" * 60)
    logger.info("开始采集: {} 个类目, 每类最多 {} 条", len(categories), max_per_category)
    logger.info("总计目标上限: {} 条", len(categories) * max_per_category)
    logger.info("=" * 60)

    scraper = CnkiScraper(browser=CnkiBrowser(headless=headless))

    total_collected = 0
    failed_categories = []

    try:
        scraper.browser.launch()

        for idx, code in enumerate(tqdm(categories, desc="类目进度", unit="cat")):
            logger.info("[{}/{}] 正在采集: {}", idx + 1, len(categories), code)

            try:
                papers = scraper.collect_category(
                    code, max_papers=max_per_category,
                    fetch_detail=fetch_detail,
                )
            except Exception:
                logger.exception("[{}] 采集异常", code)
                failed_categories.append(code)
                # 重启浏览器
                try:
                    scraper.browser.close()
                    time.sleep(2)
                    scraper.browser.launch()
                except Exception:
                    pass
                continue

            if papers:
                total_collected += len(papers)
                if save_per_category:
                    out = RAW_DIR / f"{code}.jsonl"
                    scraper.save_papers(papers, out)
                else:
                    scraper.save_papers(papers)

            # 类目间随机间隔
            time.sleep(random.uniform(5, 12))

    finally:
        scraper.browser.close()
        # 合并导出 CSV
        scraper.save_csv()

    result = {
        "total_collected": total_collected,
        "categories_attempted": len(categories),
        "failed_categories": failed_categories,
        "output_dir": str(RAW_DIR),
    }
    logger.info("=" * 60)
    logger.info("采集完成!")
    logger.info("  总条数: {}", total_collected)
    logger.info("  失败类目: {}", failed_categories)
    logger.info("  输出目录: {}", RAW_DIR)
    logger.info("=" * 60)
    return result


if __name__ == "__main__":
    # 快速测试: R51, 5 条, 含详情页提取
    logger.info("=== 快速测试: R51, 5 条, 含详情页 ===")
    run_full_crawl(
        max_per_category=5,
        categories=["R51"],
        headless=False,
        fetch_detail=True,
    )
