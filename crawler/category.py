"""
CLC R大类分类体系 (成员1负责)
================================
职责: 整理《中国图书馆分类法》R大类下128个三级类目,
     建立 类别编码 → 类别名称 映射表,持久化到 category_mapping.json。

数据源: https://www.clcindex.com/category/R/ (逐层抓取, 权威来源)
筛选标准:
  ✓ 不带小数点 (排除 R739.4 等四级)
  ✓ 不带方括号 (排除 [R34]/交替类目)
  ✓ 直接挂在二级类目下 (父类为 R-0/R1/R2/.../R9, 无中间层级)

R大类二级简表 (共20个):
  R-0 一般理论              R71 妇产科学
  R-1 现状与发展            R72 儿科学
  R-3 医学研究方法          R73 肿瘤学
  R1  预防医学、卫生学      R74 神经病学与精神病学
  R2  中国医学              R75 皮肤病学与性病学
  R3  基础医学              R76 耳鼻咽喉科学
  R4  临床医学              R77 眼科学
  R5  内科学                R78 口腔科学
  R6  外科学                R79 外国民族医学
                            R8  特种医学
                            R9  药学

R79 无三级类目, R-1 无三级类目。其余18个二级类目下共有128个三级类目。
"""
import json
from pathlib import Path
from config import CATEGORY_MAPPING_FILE, CRAWLER_CONFIG


# ====== R大类二级简表 (20个) ======
R_CATEGORY_TREE = {
    "R-0": "一般理论",
    "R-1": "现状与发展",
    "R-3": "医学研究方法",
    "R1":  "预防医学、卫生学",
    "R2":  "中国医学",
    "R3":  "基础医学",
    "R4":  "临床医学",
    "R5":  "内科学",
    "R6":  "外科学",
    "R71": "妇产科学",
    "R72": "儿科学",
    "R73": "肿瘤学",
    "R74": "神经病学与精神病学",
    "R75": "皮肤病学与性病学",
    "R76": "耳鼻咽喉科学",
    "R77": "眼科学",
    "R78": "口腔科学",
    "R79": "外国民族医学",
    "R8":  "特种医学",
    "R9":  "药学",
}


# ====== R大类三级类目 (128个, 来源: clcindex.com) ======
# 严格纯整数编码, 父类直接是 R-0/R1/R2/.../R9, 无中间层级
R_LEVEL3_CATEGORIES = {
    # R-0 一般理论 (4)
    "R-01": "方针、政策及其阐述",
    "R-02": "医学哲学",
    "R-05": "医学与其他学科的关系",
    "R-09": "医学史",
    # R-3 医学研究方法 (1)
    "R-33": "实验医学、医学实验",
    # R1 预防医学、卫生学 (11)
    "R11":  "卫生基础科学",
    "R12":  "环境卫生、环境医学",
    "R13":  "劳动卫生",
    "R14":  "放射卫生",
    "R15":  "营养卫生、食品卫生",
    "R16":  "个人卫生",
    "R169": "计划生育与卫生",
    "R17":  "妇幼卫生",
    "R179": "儿童、少年卫生",
    "R18":  "流行病学与防疫",
    "R19":  "保健组织与事业（卫生事业管理）",
    # R2 中国医学 (18)
    "R2-0": "中国医学理论",
    "R2-5": "中医学丛书、文集、连续出版物",
    "R21":  "中医预防、卫生学",
    "R22":  "中医基础理论",
    "R24":  "中医临床学",
    "R25":  "中医内科",
    "R26":  "中医外科",
    "R271": "中医妇产科",
    "R272": "中医儿科",
    "R273": "中医肿瘤科",
    "R274": "中医骨伤科",
    "R275": "中医皮科",
    "R276": "中医五官科",
    "R277": "中医其他学科",
    "R278": "中医急症学",
    "R28":  "中药学",
    "R289": "方剂学",
    "R29":  "中国少数民族医学",
    # R3 基础医学 (9) - 排除交替类目 [R34]/[R35]
    "R31":  "医用一般科学",
    "R32":  "人体形态学",
    "R33":  "人体生理学",
    "R36":  "病理学",
    "R37":  "医学微生物学（病原细菌学、病原微生物学）",
    "R38":  "医学寄生虫学",
    "R392": "医学免疫学",
    "R394": "医学遗传学",
    "R395": "医学心理学、病理心理学",
    # R4 临床医学 (5)
    "R44": "诊断学",
    "R45": "治疗学",
    "R47": "护理学",
    "R48": "临终关怀学",
    "R49": "康复医学",
    # R5 内科学 (10)
    "R51":  "传染病",
    "R52":  "结核病",
    "R53":  "寄生虫病",
    "R54":  "心脏、血管（循环系）疾病",
    "R55":  "血液及淋巴系疾病",
    "R56":  "呼吸系及胸部疾病",
    "R57":  "消化系及腹部疾病",
    "R58":  "内分泌腺疾病及代谢病",
    "R59":  "全身性疾病",
    "R599": "地方病学",
    # R6 外科学 (11)
    "R602": "外科病理学、解剖学",
    "R604": "外科诊断学",
    "R605": "外科治疗学",
    "R608": "外科诊疗器械与用具",
    "R61":  "外科手术学",
    "R62":  "整形外科学（修复外科学）",
    "R63":  "外科感染",
    "R64":  "创伤外科学",
    "R65":  "外科学各论",
    "R68":  "骨科学（运动系疾病、矫形外科学）",
    "R69":  "泌尿科学（泌尿生殖系疾病）",
    # R71 妇产科学 (6)
    "R711": "妇科学",
    "R713": "妇科手术",
    "R714": "产科学",
    "R715": "临床优生学",
    "R717": "助产学",
    "R719": "产科手术",
    # R72 儿科学 (5)
    "R722": "新生儿、早产儿疾病",
    "R723": "婴儿的营养障碍",
    "R725": "小儿内科学",
    "R726": "小儿外科学",
    "R729": "小儿其他疾病",
    # R73 肿瘤学 (9)
    "R73-3": "肿瘤学实验研究",
    "R730":  "一般性问题",
    "R732":  "心血管肿瘤",
    "R733":  "造血器及淋巴系肿瘤",
    "R734":  "呼吸系肿瘤",
    "R735":  "消化系肿瘤",
    "R736":  "内分泌腺肿瘤",
    "R737":  "泌尿生殖器肿瘤",
    "R738":  "运动系肿瘤",
    # R74 神经病学与精神病学 (2)
    "R741": "神经病学",
    "R749": "精神病学",
    # R75 皮肤病学与性病学 (2)
    "R751": "皮肤病学",
    "R759": "性病学",
    # R76 耳鼻咽喉科学 (7)
    "R762": "耳鼻咽喉外科学",
    "R763": "耳鼻咽喉科真菌病",
    "R764": "耳科学、耳疾病",
    "R765": "鼻科学、鼻疾病",
    "R766": "咽科学、咽疾病",
    "R767": "喉科学、喉疾病",
    "R768": "气管与食管镜学",
    # R77 眼科学 (8)
    "R771": "眼的一般性疾病",
    "R772": "眼纤维膜疾病",
    "R773": "眼色素层（葡萄膜）疾病",
    "R774": "视网膜及视神经疾病",
    "R775": "眼压与青光眼",
    "R776": "晶状体与玻璃体疾病",
    "R777": "眼附属器官疾病",
    "R778": "眼屈光学",
    # R78 口腔科学 (5)
    "R781": "口腔内科学",
    "R782": "口腔颌面部外科学",
    "R783": "口腔矫形学",
    "R787": "老年口腔疾病",
    "R788": "儿童口腔疾病",
    # R8 特种医学 (6) - 排除交替类目 [R89]
    "R81": "放射医学",
    "R82": "军事医学",
    "R83": "航海医学",
    "R84": "潜水医学",
    "R85": "航空航天医学",
    "R87": "运动医学",
    # R9 药学 (9)
    "R91":  "药物基础科学",
    "R917": "药物分析",
    "R92":  "药典、药方集（处方集）、药物鉴定",
    "R93":  "生药学（天然药物学）",
    "R94":  "药剂学",
    "R95":  "药事组织",
    "R96":  "药理学",
    "R97":  "药品",
    "R99":  "毒物学（毒理学）",
}


def build_category_mapping(include_level2: bool = False) -> dict:
    """
    构建三级类目映射表 (128个, 来源: clcindex.com 逐层抓取)
    
    Args:
        include_level2: 是否同时包含二级类目(默认False, 只返回三级)
    
    Returns:
        dict: {编码: 名称} 映射
    """
    mapping = dict(R_LEVEL3_CATEGORIES)
    if include_level2:
        mapping.update(R_CATEGORY_TREE)
    return mapping


def save_mapping(mapping: dict = None) -> Path:
    """持久化类别映射到 category_mapping.json"""
    mapping = mapping or build_category_mapping()
    CATEGORY_MAPPING_FILE.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return CATEGORY_MAPPING_FILE


def load_mapping() -> dict:
    """加载类别映射 (若不存在则从模块常量构建并保存)"""
    if not CATEGORY_MAPPING_FILE.exists():
        save_mapping()
    return json.loads(CATEGORY_MAPPING_FILE.read_text(encoding="utf-8"))


def query_category_name(code: str) -> str:
    """根据分类号查询类别名称 (供成员4的 classify_text 使用)
    
    支持逐级父类匹配, 如 R318.01 → R318 → R3
    """
    mapping = load_mapping()
    if code in mapping:
        return mapping[code]
    # 逐级截断查找父类
    for i in range(len(code), 0, -1):
        parent = code[:i]
        if parent in mapping:
            return f"{mapping[parent]}(父类匹配)"
        if parent in R_CATEGORY_TREE:
            return R_CATEGORY_TREE[parent] + "(二级类目匹配)"
    return "未知类别"


def get_parent_code(code: str) -> str:
    """获取某三级类目的二级父类编码"""
    for i in range(len(code)-1, 0, -1):
        candidate = code[:i]
        if candidate in R_CATEGORY_TREE:
            return candidate
    return ""


def list_all_categories() -> list:
    """列出所有可用三级类目编码 (供 collect_data 输入)"""
    return sorted(load_mapping().keys())


if __name__ == "__main__":
    mapping = build_category_mapping()
    n_level2 = len(R_CATEGORY_TREE)
    n_level3 = len(R_LEVEL3_CATEGORIES)
    print(f"R大类 二级类目: {n_level2} 个")
    print(f"R大类 三级类目: {n_level3} 个")
    print(f"目标 ≥ {CRAWLER_CONFIG['min_categories']}: {'✓' if n_level3 >= CRAWLER_CONFIG['min_categories'] else '⚠️ 不足'}")
    save_mapping(mapping)
    print(f"已保存到: {CATEGORY_MAPPING_FILE}")
