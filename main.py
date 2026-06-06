# main.py
import os
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

# Add src folder to Python system path to load custom transformations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import transformations as T

def main():
    # Initialize Spark Session configured for local development
    spark = SparkSession.builder \
        .appName("MusicAnalysis") \
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

    # Load datasets
    print("Ingesting CSV datasets...")
    logs_df = spark.read.csv("listening_logs.csv", header=True, schema=logs_schema)
    metadata_df = spark.read.csv("songs_metadata.csv", header=True, schema=metadata_schema)

    # Performance optimization: Broadcast small metadata table to prevent shuffling logs
    joined_df = logs_df.join(F.broadcast(metadata_df), on="song_id", how="inner")

    # Task 1: User Favorite Genres
    fav_genres_df = T.get_user_favorite_genres(joined_df)
    print("\n=============================================")
    print("📊 TASK 1: User Favorite Genres (Sample)")
    print("=============================================")
    fav_genres_df.show(5)

    # Task 2: Average Listen Time
    avg_listen_time_df = T.get_average_listen_time(joined_df)
    print("\n=============================================")
    print("⏱️ TASK 2: Average Listen Time (Sample)")
    print("=============================================")
    avg_listen_time_df.show(5)

    # Task 3: Genre Loyalty Scores - Top 10
    genre_loyalty_df = T.get_genre_loyalty_scores(joined_df)
    print("\n=============================================")
    print("🏆 TASK 3: Genre Loyalty Scores - Top 10")
    print("=============================================")
    genre_loyalty_df.show(10)

    # Task 4: Night Owl Users
    night_owls_df = T.get_night_owl_users(joined_df)
    print("\n=============================================")
    print("🦉 TASK 4: Night Owl Users (Sample)")
    print("=============================================")
    night_owls_df.show(5)

    # Ensure the outputs/ directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save outputs as single CSV files to outputs/ directory
    print("\n💾 Saving results to outputs/ directory...")
    fav_genres_df.toPandas().to_csv("outputs/user_favorite_genres.csv", index=False)
    avg_listen_time_df.toPandas().to_csv("outputs/avg_listen_time.csv", index=False)
    genre_loyalty_df.toPandas().to_csv("outputs/user_genre_loyalty_top10.csv", index=False)
    night_owls_df.toPandas().to_csv("outputs/night_owl_users.csv", index=False)
    print("Done! All results written successfully.")

    # Stop the Spark session cleanly
    spark.stop()

if __name__ == "__main__":
    main()
