# 成员2工作总结与项目交接

更新：2026-09-03，本文数据已切换至 raw1/raw2/raw3 新交付版本。
详细核验结果与补采建议见 `preprocess/DATASET_REPORT_2026-09-03.md`。
旧说明文档的有效内容已融入当前版本；历史备份目录 `data/archive/` 已按用户要求删除。
旧三类别数据（包括新版未收录的2篇R52历史记录）不再有项目内归档副本，不能通过该归档回退；当前6162条、124类数据集不受影响。

## 1. 已完成工作

成员2负责的数据清洗与数据集构建流程已经实现并通过现有真实数据验证：

1. 原始 JSONL 读取与格式校验；
2. HTML、Unicode、不可见字符和空白清洗；
3. 关键词拆分、去重与列表化；
4. 日期、日期时间以及 YYYYMM/YYYYMMDD 统一为四位年份，同时保留 `year_raw`；
5. 使用 `category_mapping.json` 校验并映射标准标签；
6. DOI、“标准化标题 + 来源”以及完全相同模型输入联合去重；冲突标签全部隔离，非冲突组保留并合并最完整记录；
7. 构造“标题 + 关键词 + 摘要”模型输入；
8. 输出缺摘要、缺关键词、缺 DOI、疑似截断摘要等质量标记；
9. 按类别确定性分层划分训练集、验证集和测试集；
10. 生成稳定的标签编号、类别名称映射和类别权重；
11. 支持仅训练集过采样；
12. 生成清洗、统计和划分报告；
13. 接通 Agent 的 `clean_dataset` 工具；
14. 提供一键运行脚本、18项自动测试和自动验收；
15. 导入交付包前逐文件核验、备份旧数据，记录来源与SHA-256；
16. 修复HTML正则误删 P<0.05 与 P>0.05 之间文本的问题；
17. 输出逐类别质量报告、冲突复核清单及版本指纹。

## 2. 当前真实数据结果

- 原始数据：6379条、128类，即127类各50条，加R31的29条；
- raw2的22个文件、raw3的21个文件与raw1对应文件完全一致，未重复追加；
- 隔离标签冲突：60组、216条；合并同标签等价模型输入：1条；
- 数据守恒：6379 = 6162 + 216 + 1；
- 清洗后：6162条、124类，R31仍为29条；
- 暂不用于训练：R-01、R-02、R-05、R-09（样本全部存在跨类冲突）；
- 训练集：4310条；验证集：618条；测试集：1234条；
- 三个集合均覆盖124类；ID、输入文本、DOI、标题+来源均无跨集合交叉；
- 缺摘要2778条，缺关键词3498条，缺DOI 4647条，缺来源301条；
- 年份全部可用；2296条原始年份格式得到统一，其中50条为紧凑年月格式；
- 疑似摘要截断1121条；43个类别全部缺摘要，见统计文件；
- 类别不平衡比例1.724138，默认不重采样；权重由成员3按需启用；
- 满足“至少100类、3000条”的数量条件，但不等于标签质量已核实或模型效果达标。

## 3. 运行方式

```bash
python run_preprocess.py
python -m preprocess.validate
python -m unittest discover -s tests -v
```

每次成员1增加或更新 `data/raw/R*.jsonl` 后，重新执行第一条命令即可覆盖生成最新数据集。
默认参数是训练/验证/测试=70%/10%/20%，随机种子固定为42。

## 4. 成员2相关文件

- `run_preprocess.py`：一键执行清洗、统计和划分。
- `preprocess/clean.py`：清洗、字段规范化、标签校验、联合去重、文本构造、质量标记。
- `preprocess/split.py`：分层划分、标签编号、类别权重、训练集过采样。
- `preprocess/statistics.py`：类别分布、缺失率、文本长度和质量统计。
- `preprocess/import_raw.py`：交付包核验、同名文件检查、旧版本归档和原始文件导入。
- `preprocess/validate.py`：原始文件指纹、产物版本、数据守恒、集合泄漏和标签一致性验收。
- `preprocess/__init__.py`：向其他模块导出成员2公共函数。
- `preprocess/README.md`：使用方法、数据契约和输出说明。
- `agent/tools/clean_dataset.py`：成员4可以调用的预处理 Agent 工具。
- `tests/test_preprocess.py`：18项清洗、导入、冲突隔离、分层划分、平衡和验收回归测试。
- `preprocess/DATASET_REPORT_2026-09-03.md`：本批交付数量核对、质量问题和成员1/3/4/5交接说明。

## 5. 成员1及原始数据文件

- `run_crawl.py`：知网采集命令行入口。
- `crawler/cnki.py`：浏览器控制、结果列表解析、详情提取、断点续采和原始数据保存。
- `crawler/category.py`：CLC R类分类体系、类别名称查询和映射生成。
- `crawler/__init__.py`：导出爬虫公共接口。
- `category_mapping.json`：128个目标类别的编码与名称映射。
- `data/raw/R51.jsonl`：R51传染病原始数据。
- `data/raw/R52.jsonl`：R52结核病原始数据。
- `data/raw/R54.jsonl`：R54心血管疾病原始数据。
- `data/raw/R*.jsonl`：现在共有128个类别文件；R-33与R33是不同类别，文件名及标签不会混淆。
- `data/raw/cnki_papers.csv`：所有分类 JSONL 的查看/交付副本；预处理不会读取它，防止数据翻倍。

## 6. 生成的数据文件

- `data/processed/cleaned.jsonl`：清洗、合并、去重后的完整标准数据。
- `data/processed/rejected.jsonl`：216条冲突记录，包含完整清洗字段、原始文件及行号；隔离不是删除原始数据。
- `data/processed/label_conflicts.jsonl`：60个冲突组的标题、候选标签与定位信息，交给成员1复核。
- `data/processed/duplicate_groups.jsonl`：同标签去重审计，记录保留ID和所有来源。
- `data/processed/quality_report.json`：原始数、清洗数、去重数和异常数量。
- `data/processed/statistics.json`：成员5可直接展示的数据统计。
- `data/processed/split_report.json`：划分参数和每类在三个集合中的数量。
- `data/processed/validation_report.json`：自动验收结果、集合交叉数、数量目标及版本指纹。
- `data/import_manifest.json`：本次导入所有源路径、文件哈希、复制/替换操作和备份位置。
- 历史 `data/archive/`：已按用户要求删除，不再提供旧版本恢复；导入清单中的备份路径仅为历史记录。
- `data/train.jsonl`：成员3训练使用。
- `data/val.jsonl`：成员3调参与选择模型使用。
- `data/test.jsonl`：成员3最终评估使用，不能参与训练和调参。
- `data/labels.json`：标签编号、编码名称和训练类别权重。

## 7. 其他项目文件

- `config.py`：全项目路径以及爬虫、模型、Agent、Web公共配置。
- `requirements.txt`：各模块依赖列表；成员2核心流程目前只使用标准库。
- `.gitignore`：排除虚拟环境、缓存、日志和大模型权重。
- `model/train.py`：成员3训练入口骨架。
- `model/evaluate.py`：成员3模型评估骨架。
- `model/classifier.py`：成员3提供给Agent的统一推理接口骨架。
- `model/__init__.py`：导出模型公共接口。
- `agent/agent.py`：Agent计划与工具执行入口，目前是关键词规则版本。
- `agent/graph.py`：成员4的 LangGraph 工作流骨架。
- `agent/prompts/__init__.py`：成员4的提示词模板。
- `agent/tools/base.py`：所有Agent工具共享的结果、步骤和日志结构。
- `agent/tools/__init__.py`：Agent工具注册表。
- `agent/tools/collect_data.py`：采集工具包装骨架。
- `agent/tools/train_model.py`：训练工具包装骨架。
- `agent/tools/classify_text.py`：分类推理工具包装骨架。
- `agent/tools/serve_web.py`：Web服务工具包装骨架。
- `backend/main.py`：成员5的 FastAPI 路由骨架。
- `backend/__init__.py`、`agent/__init__.py`：Python包及公共接口导出。
- `frontend/.gitkeep`：前端目录占位。
- `model/checkpoints/.gitkeep`：模型权重目录占位。
- `results/.gitkeep`：指标和混淆矩阵目录占位。
- `presentation/.gitkeep`：PPT及演示资料目录占位。
- `.venv/`、`.deps/`：本机依赖环境，不是业务代码。
- `__pycache__/`、`*.pyc`：Python运行缓存，不应手工维护。

## 8. 后续成员注意事项

### 成员1

1. 当前详情页 `classification` 全为空，现有 `label` 来自检索条件 `clc_code`，属于弱标签；建议补抓真实分类号或抽样人工核验。
2. 优先复核R-01/R-02/R-05/R-09的检索结果：它们高度重复且类别不同；不能擅自选一个标签。
3. 继续采集时每个类别应保证去重后仍有足够样本，不要只看写入行数。
4. 当前采集端 UID 会受年份格式影响，预处理已能纠正重复，但采集端仍建议统一年份后再生成 UID。
5. 补抓43个无摘要类别的详情页，并核查1121条疑似截断摘要；确认检索标签与真实分类号是否一致。

### 成员3

1. 训练字段使用 `text`，标签使用 `label_id`；类别解释读取 `data/labels.json`。
2. 模型输出类别数必须取 `len(label2id)=124`，不能取category_mapping的128，也不能继续使用旧三类标签编号。
3. 不要用 `source`、作者或URL作为训练文本，避免模型记忆期刊等捷径特征。
4. `quality_flags` 可用于质量分析；直接删除全部无摘要样本会丢失43类，只能在明确限定类别的对照实验中使用。
5. `class_weights` 只基于训练集计算，程序不自动修改训练损失；如选过采样，不建议再同时启用同一套补偿权重。
6. 旧模型与旧标签文件不能混用。每次扩大类别/重建划分后需重新训练并把labels及dataset_sha256随权重保存。

### 成员4、成员5

1. Agent可通过 `CleanDatasetTool().run()` 或自然语言“清洗数据”触发完整流程。
2. `/api/statistics` 后续可以直接读取 `data/processed/statistics.json`。
3. 质量报告和划分报告均为普通JSON，适合直接在前端展示。
4. 前端分别展示“原始128类”和“可训练124类”，并解释4个隔离类别；不能宣称支持预测全部128类。

## 9. 尚不能由清洗模块自动解决的问题

清洗模块不会伪造缺失摘要、真实分类号、机构和期刊卷期页码。上述字段只能由成员1重新采集或人工补充。当前处理方式是保留可训练内容并明确标记质量问题，保证流程可运行且结果可追溯。
