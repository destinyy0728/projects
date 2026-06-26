# VED EV-Analytics: 新能源汽车海量轨迹与能耗分析系统

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PySpark](https://img.shields.io/badge/PySpark-Big_Data_ETL-E25A1C)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-1E88E5)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 项目背景与业务价值 (Business Value)
本项目基于密歇根大学开源的 **Vehicle Energy Dataset (VED)**，旨在解决新能源车企在车联网数据应用中的两个核心痛点：
1. **千人千面的驾驶行为画像**：通过分析用户的急加速、急刹车等行为，对驾驶偏好进行无监督聚类，为车险定价（UBI）或车辆能耗优化提供数据支撑。
2. **表显剩余续航 (GOM) 动态校准**：结合实时环境温度、车速及个人驾驶习惯，利用机器学习算法进行高精度单次行程能耗预估，缓解用户的“里程焦虑”。

## 🏗️ 核心架构 (Technical Architecture)
本项目采用真实工业界**“离线大数据处理 + 在线轻量级推理”**的解耦架构，独立处理超过 **2,300 万行**的高频时序传感器数据。

* **[Phase 1] Cloud ETL (PySpark)**
  * 在云端节点读取 GB 级的原始高频轨迹 CSV 数据。
  * 利用 Spark 窗口函数 (Window Functions) 计算车辆秒级加速度，提取急加/减速业务特征。
  * 降维压缩，将千万级时序流水账聚合为轻量级的 `Driver Profile` (Parquet 格式) 画像矩阵。
* **[Phase 2] ML Modeling (XGBoost & Scikit-learn)**
  * 使用 `K-Means` 算法对底层画像特征进行聚类分析。
  * 训练 `XGBoost` 树模型，基于多维时空和行为特征实现单次行程的等效能耗回归预测。
* **[Phase 3] Interactive BI Dashboard (Streamlit)**
  * 采用冷峻工业风 UI，零延迟读取 Parquet 宽表。
  * 封装 ML 推理引擎，业务端可直接拖动环境变量，实时输出等效电耗估算与百公里成本。

## 🚀 核心亮点 (Key Highlights)
* **千万级数据处理能力**：利用分布式计算思维克服单机内存瓶颈。
* **端到端工程闭环**：从最底层的脏数据清洗、特征工程，一直贯穿到前端交互全栈开发。
* **极简前端渲染**：摒弃传统 Pandas 读取大文件的性能瓶颈，采用 Parquet 列式存储保障前端秒级刷新。

## 📂 仓库结构 (Repository Structure)
```text
VED_project/
├── app.py                      # Streamlit 前端交互看板与在线推理逻辑
├── train_model.py              # XGBoost 模型离线训练与特征构造脚本
├── driver_profiles.parquet     # 经 PySpark 降维提取后的核心业务特征宽表
├── xgb_model.pkl               # 序列化保存的 XGBoost 回归预测模型
├── VED_Combustion_Master_Cleaned.csv # 原始高频数据集快照 (抽样展示用)
└── README.md                   # 项目说明文档
```
*(注：为保证 Git 仓库的轻量化，完整的云端 PySpark ETL 处理源码与探索性分析 (EDA) 笔记另附于 GitHub Gist 或 Kaggle 链接中。)*

## ⚙️ 快速启动 (Quick Start)

**1. 克隆仓库与安装依赖**
```bash
git clone https://github.com/your-username/ved-ev-analytics.git
cd ved-ev-analytics
pip install -r requirements.txt
```

**2. 启动工业级数据看板**
```bash
streamlit run app.py
```
*(服务将在本地 `http://localhost:8501` 启动)*