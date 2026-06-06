from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
import transformations as T

def main():
    # Initialize Spark Session configured for local development
    spark = SparkSession.builder \
        .appName("MusicStreamingAnalytics") \
        .master("local[*]") \
        .getOrCreate()
        
    # Suppress verbose info logs to keep terminal output clean
    spark.sparkContext.setLogLevel("WARN")

    # Define strict Schemas matching datagen.py output
    logs_schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("song_id", StringType(), False),
        StructField("timestamp", TimestampType(), True),
        StructField("duration_sec", IntegerType(), True)
    ])

    metadata_schema = StructType([
        StructField("song_id", StringType(), False),
        StructField("title", StringType(), True),
        StructField("artist", StringType(), True),
        StructField("genre", StringType(), True),
        StructField("mood", StringType(), True)
    ])

    print("📥 Ingesting CSV datasets...")
    logs_df = spark.read.csv("listening_logs.csv", header=True, schema=logs_schema)
    metadata_df = spark.read.csv("songs_metadata.csv", header=True, schema=metadata_schema)

    # Performance optimization: Broadcast small metadata table to prevent shuffling logs
    joined_df = logs_df.join(F.broadcast(metadata_df), on="song_id", how="inner")

    # ---- RUN TASKS ----
    print("\n=============================================")
    print("📊 TASK 1: User Favorite Genres (Sample)")
    print("=============================================")
    T.get_user_favorite_genres(joined_df).show(5)

    print("\n=============================================")
    print("⏱️ TASK 2: Average Listen Time (Sample)")
    print("=============================================")
    T.get_average_listen_time(joined_df).show(5)

    print("\n=============================================")
    print("🏆 TASK 3: Genre Loyalty Scores - Top 10")
    print("=============================================")
    T.get_genre_loyalty_scores(joined_df).show(10)

    print("\n=============================================")
    print("🦉 TASK 4: Night Owl Users (Sample)")
    print("=============================================")
    T.get_night_owl_users(joined_df).show(5)

    # Stop the Spark Session cleanly
    spark.stop()

if __name__ == "__main__":
    main()