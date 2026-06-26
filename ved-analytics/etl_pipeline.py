import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def create_spark_session():
    """初始化并配置 Spark 引擎"""
    return SparkSession.builder \
        .appName("VED_Trajectory_ETL") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

def process_ved_data(spark, input_path, output_path):
    """
    核心 ETL 流水线: 读取原始数据 -> 异常值清洗 -> 复杂时序特征提取 -> 画像降维 -> 导出
    """
    print(f"🔥 [ETL] 开始读取海量轨迹数据: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # 1. 数据清洗 (Data Cleaning)
    # 过滤掉缺少关键字段(车速、时间戳)的脏数据
    df_clean = df.filter(F.col("Vehicle_Speed_km_per_h").isNotNull() & F.col("Timestampms").isNotNull())

    # 2. 时序特征工程 (Time-Series Feature Engineering)
    print("⏳ [ETL] 正在执行分布式窗口运算计算加速度...")
    window_spec = Window.partitionBy("VehId", "Trip").orderBy("Timestampms")

    df_features = df_clean.withColumn("prev_time", F.lag("Timestampms").over(window_spec)) \
                          .withColumn("prev_speed", F.lag("Vehicle_Speed_km_per_h").over(window_spec)) \
                          .withColumn("delta_t_sec", (F.col("Timestampms") - F.col("prev_time")) / 1000.0) \
                          .withColumn("delta_v_mps", (F.col("Vehicle_Speed_km_per_h") - F.col("prev_speed")) / 3.6)

    # 计算加速度并过滤异常跳跃点
    df_features = df_features.filter((F.col("delta_t_sec") > 0) & (F.col("delta_t_sec") < 100)) \
                             .withColumn("acceleration", F.col("delta_v_mps") / F.col("delta_t_sec"))

    # 业务规则打标：定义急加速与急刹车阈值 (2.5 m/s^2)
    df_features = df_features.withColumn("is_rapid_accel", F.when(F.col("acceleration") > 2.5, 1).otherwise(0)) \
                             .withColumn("is_harsh_brake", F.when(F.col("acceleration") < -2.5, 1).otherwise(0))

    # 3. 数据降维聚合 (Data Aggregation & Dimensionality Reduction)
    print("📉 [ETL] 正在按车辆实体聚合驾驶行为画像...")
    driver_profile = df_features.groupBy("VehId").agg(
        F.countDistinct("Trip").alias("total_trips"),
        F.round(F.avg("Vehicle_Speed_km_per_h"), 2).alias("avg_speed_kmh"),
        F.sum("is_rapid_accel").alias("rapid_accel_count"),
        F.sum("is_harsh_brake").alias("harsh_brake_count"),
        F.round(F.avg("OAT_DegC"), 2).alias("avg_out_temp"),
        F.round(F.sum("MAF_g_per_sec"), 2).alias("total_energy_proxy")
    )

    # 消除行程长短带来的频次绝对值误差，计算“单次行程平均发生率”
    driver_profile = driver_profile.withColumn("rapid_accel_per_trip", F.round(F.col("rapid_accel_count") / F.col("total_trips"), 2)) \
                                   .withColumn("harsh_brake_per_trip", F.round(F.col("harsh_brake_count") / F.col("total_trips"), 2))

    # 4. 数据导出 (Export)
    print(f"💾 [ETL] 画像数据持久化至 Parquet: {output_path}")
    driver_profile.repartition(1).write.mode("overwrite").parquet(output_path)
    print("✅ [ETL] 全部流水线执行完毕！")

if __name__ == "__main__":
    # 模拟云端运行入口
    spark = create_spark_session()

    # 路径配置 (适配云端/集群环境)
    INPUT_DATA_PATH = "/path/to/raw/VED_Data/**/*.csv"
    OUTPUT_DATA_PATH = "./driver_profiles_output"

    # 在实际集群中取消下方注释即可运行
    # process_ved_data(spark, INPUT_DATA_PATH, OUTPUT_DATA_PATH)

    print("ETL 脚本就绪。由于本地内存限制，本脚本主要在集群或 Kaggle 云端执行。")
