# Skin Diseases 皮肤病智能诊断系统

基于深度学习的皮肤病智能诊断系统，集成了图像分类、多Agent智能诊断和RAG知识增强三大核心功能。

## 功能特点

- **图像分类**: 支持23类皮肤病的识别与分类
- **多Agent智能诊断**: 基于大语言模型的症状分析、图像诊断、治疗建议
- **RAG知识增强**: 检索增强生成，提供专业的医学知识库

## 技术栈

- **深度学习**: PyTorch, torchvision, EfficientNet
- **向量数据库**: ChromaDB
- **LLM集成**: LangChain

## 项目结构

```
源代码/
├── main.py                    # 模型训练主入口
├── train_validation.py        # 训练与验证逻辑
├── evaluate.py               # 模型评估脚本
├── test.py                   # 测试脚本
├── dataset_split.py          # 数据集划分
├── requirements.txt          # Python依赖
│
├── model/                    # 模型定义
│   ├── PanDerm.py           # EfficientNet-B3皮肤病分类模型
│   ├── ResNet50.py          # ResNet50分类模型
│   ├── custom_skin_net.py   # 自定义皮肤病网络(CBAM)
│   ├── ConvNeXtTiny.py      # ConvNeXt-Tiny分类模型
│   └── factory.py           # 模型工厂
│
├── utils/                    # 工具函数
│   ├── dataset.py           # 数据集加载与增强
│   ├── arguments.py         # 命令行参数解析
│   ├── config_handler.py    # YAML配置加载器
│   ├── lr_policy.py         # 学习率调度
│   ├── optimizer_Adam.py    # Adam优化器
│   ├── outputwriter.py      # 模型保存
│   ├── writer.py           # TensorBoard日志
│   ├── logger.py           # 日志管理
│   ├── prompt_loader.py    # 提示词加载
│   ├── path_tool.py        # 路径工具
│   ├── file_handler.py     # 文件处理
│   └── first_order_oracle.py # 性能评估Oracle
│
├── config/                   # 配置文件
│   ├── model.yml            # 模型配置(23类皮肤病)
│   ├── test_evaluate.yml    # 测试评估配置
│   ├── rag.yml             # RAG模块配置
│   ├── prompts.yml         # 提示词配置
│   ├── agent.yml           # Agent配置
│   ├── chroma.yml          # 向量数据库配置
│   ├── default.yml         # 默认配置
│   └── instructions.yml    # 指令配置
│
├── data/                     # 知识库数据
│   ├── disease_info.txt     # 皮肤病基础知识
│   ├── common_qa.txt        # 常见问答
│   ├── prevention_tips.txt  # 预防建议
│   ├── medication_guide.txt # 用药指南
│   ├── differential_diagnosis.txt # 鉴别诊断
│   └── medical_advice.txt   # 医疗建议
│
├── prompts/                  # 提示词文件
│   ├── final_response_prompt.txt   # 最终响应提示
│   ├── rag_summarize.txt           # RAG总结提示
│   ├── skinderm_llm_decision_prompt.txt  # LLM决策提示
│   └── task_decision_prompt.txt    # 任务决策提示
│
├── agent/                    # Agent模块
│   ├── multi_agent_manager.py     # 多Agent管理器
│   ├── symptom_agent.py            # 症状分析Agent
│   ├── image_agent.py              # 图像诊断Agent
│   ├── treatment_agent.py          # 治疗建议Agent
│   └── tools/
│       ├── agent_tools.py         # Agent工具函数
│       └── middleware.py          # 中间件
│
├── rag/                      # RAG模块
│   ├── enhanced_rag.py      # 增强RAG服务
│   └── vector_store.py      # 向量存储服务
│
├── backend/                  # 后端API
│   └── main.py             # FastAPI主入口
│
├── frontend-react/           # 前端界面
│   └── (React + Vite + Tailwind CSS)
│
└── variables/               # 变量定义(模型权重等)
```

## 支持的皮肤病类别 (23类)

1. 痤疮和酒渣鼻 (Acne and Rosacea)
2. 光化性角化病和基底细胞癌 (Actinic Keratosis Basal Cell Carcinoma)
3. 特应性皮炎 (Atopic Dermatitis)
4. 大疱性疾病 (Bullous Disease)
5. 蜂窝织炎和脓疱病 (Cellulitis Impetigo)
6. 湿疹 (Eczema)
7. 发疹和药物反应 (Exanthems and Drug Eruptions)
8. 脱发 (Hair Loss)
9. 疱疹、HPV及其他性病 (Herpes HPV and other STDs)
10. 色素性疾病 (Light Diseases and Disorders of Pigmentation)
11. 狼疮及结缔组织病 (Lupus and other Connective Tissue diseases)
12. 黑色素瘤、皮肤癌与痣 (Melanoma Skin Cancer Nevi)
13. 指甲疾病 (Nail Fungus and other Nail Disease)
14. 毒葛皮炎 (Poison Ivy and other Contact Dermatitis)
15. 银屑病与扁平苔藓 (Psoriasis Lichen Planus)
16. 疥疮、莱姆病及寄生虫感染 (Scabies Lyme Disease)
17. 脂溢性角化病及良性肿瘤 (Seborrheic Keratoses)
18. 系统性疾病 (Systemic Disease)
19. 真菌感染 (Tinea Ringworm Candidiasis)
20. 荨麻疹 (Urticaria Hives)
21. 血管瘤 (Vascular Tumors)
22. 血管炎 (Vasculitis)
23. 疣、传染性软疣及病毒感染 (Warts Molluscum)

## 运行环境

Python: 3.11

操作系统:Windows 11


## 安装

```bash
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 使用方法

### 模型训练
```bash
python main.py --val True
python main.py --model convnext_tiny --val True
python main.py --model convnext_tiny --loss class_balanced --cb-loss-type focal --cb-beta 0.9999 --cb-gamma 2.0 --val True
```

Optimizer examples:

```bash
python main.py --optimizer SGD --lr 0.1 --optimizer-param momentum=0.9 --optimizer-param nesterov=true
python main.py --optimizer AdamW --optimizer-params "{betas: [0.9, 0.999], eps: 1.0e-8}"
python main.py --optimizer RMSprop --optimizer-param alpha=0.95 --optimizer-param momentum=0.9
```

YAML config example:

```yaml
optimizer: SGD
lr: 0.1
weight-decay: 0.0001
optimizer-params:
  momentum: 0.9
  nesterov: true
```

普通交叉熵权重保存到 `variables/<model>/`；Class-Balanced Loss 权重保存到 `variables/<model>/class_balanced_<type>/`。
训练过程会在同一目录下实时覆盖更新 `training_metrics.png`，并同步写入 `training_metrics.csv`。
最佳验证指标会同步覆盖写入 `best_metrics.json`，并在指标图中用红色虚线标出 best epoch。

### 模型评估
```bash
python evaluate.py
python evaluate.py --model convnext_tiny
```

### 模型效果测试
```bash
python test.py
```

### 启动前端
先进入前端目录
```bash
npm install
npm run dev
```

### 启动后端
先进入后端目录
```bash
$env:DASHSCOPE_API_KEY="sk-b371ebde92284d9ebf00b32645ea6edd"
python main.py
```

## 模型权重

训练好的模型保存在 `variables/` 目录下，可通过 `--resume` 参数恢复训练。
