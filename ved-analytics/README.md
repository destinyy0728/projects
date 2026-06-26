# VED EV-Analytics: 新能源汽车轨迹与能耗分析系统

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PySpark](https://img.shields.io/badge/PySpark-ETL-E25A1C)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-1E88E5)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)

## 项目背景与业务价值
本项目基于开源的 Vehicle Energy Dataset (VED)，旨在解决新能源车企业务中的两个问题：
1. 驾驶行为画像：通过分析用户的急加速、急刹车等行为，对驾驶偏好进行无监督聚类，为车险定价或车辆能耗优化提供数据支撑。
2. 表显剩余续航动态校准：结合实时环境温度、车速及个人驾驶习惯，利用机器学习算法进行高精度单次行程能耗预估。

## 核心架构
本项目采用离线大数据处理结合在线轻量级推理的架构，独立处理超过 2,300 万行的高频时序传感器数据。

* Phase 1: Cloud ETL (PySpark)
  * 在云端节点读取原始高频轨迹 CSV 数据。
  * 利用 Spark 窗口函数计算车辆秒级加速度，提取急加减速特征。
  * 降维压缩，将时序流水账聚合为轻量级的画像矩阵 (Parquet 格式)。

* Phase 2: ML Modeling (XGBoost & Scikit-learn)
  * 使用 K-Means 算法对底层画像特征进行聚类分析。
  * 训练 XGBoost 树模型，基于多维时空和行为特征实现单次行程的等效能耗回归预测。

* Phase 3: Interactive Dashboard (Streamlit)
  * 构建交互式数据看板，高效读取 Parquet 宽表。
  * 封装 ML 推理引擎，业务端可调整环境变量输入，实时输出等效电耗估算与百公里成本。

## 核心亮点
* 千万级数据处理能力：利用分布式计算框架处理 2300 万行时序数据，克服单机内存瓶颈。
* 端到端工程实现：涵盖底层数据清洗、特征工程、模型训练及前端交互的全流程开发。
* 高效前端渲染：采用 Parquet 列式存储代替全量 CSV 读取，保障前端交互的低延迟响应。

## 仓库结构
```text
ved-analytics/
├── app.py                      # Streamlit 前端交互看板与推理逻辑
├── train_model.py              # XGBoost 模型离线训练脚本
├── etl_pipeline.py             # PySpark 数据清洗与特征提取脚本
├── driver_profiles.parquet     # 提取后的聚合特征宽表
├── xgb_model.pkl               # 训练好的 XGBoost 预测模型
├── VED_Combustion_Master_Cleaned.csv # 原始高频数据集抽样快照
├── requirements.txt            # 项目依赖清单
└── README.md                   # 项目说明文档
```

## 快速启动

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 启动数据看板
```bash
streamlit run app.py
```