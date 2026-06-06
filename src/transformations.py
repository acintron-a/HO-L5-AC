from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F

def get_user_favorite_genres(joined_df: DataFrame) -> DataFrame:
    """
    Task 1: User Favorite Genres
    Identifies the most listened genre for each user.
    """
    # Count plays per user per genre
    genre_counts = joined_df.groupBy("user_id", "genre").count()
    
    # Define window to rank genres for each user based on play count
    window_spec = Window.partitionBy("user_id").orderBy(F.col("count").desc())
    
    # Filter for the top rank (rank == 1)
    return genre_counts.withColumn("rank", F.row_number().over(window_spec)) \
        .filter(F.col("rank") == 1) \
        .select("user_id", F.col("genre").alias("favorite_genre"), F.col("count").alias("genre_play_count"))


def get_average_listen_time(joined_df: DataFrame) -> DataFrame:
    """
    Task 2: Average Listen Time
    Calculates the average listening duration in seconds for each user.
    """
    return joined_df.groupBy("user_id") \
        .agg(F.round(F.avg("duration_sec"), 2).alias("avg_duration_sec"))


def get_genre_loyalty_scores(joined_df: DataFrame) -> DataFrame:
    """
    Task 3: Genre Loyalty Scores - Top 10
    Calculates the percentage of total listens that belong to their top genre.
    """
    # 1. Get total plays per user using a Window total (prevents early aggregation collapse)
    total_window = Window.partitionBy("user_id")
    df_with_totals = joined_df.withColumn("total_plays", F.count("song_id").over(total_window))
    
    # 2. Group by user and genre to get specific counts alongside the pre-calculated total
    genre_counts = df_with_totals.groupBy("user_id", "genre", "total_plays").count()
    
    # 3. Rank genres per user to find their top genre
    rank_window = Window.partitionBy("user_id").orderBy(F.col("count").desc())
    
    # 4. Filter for top genre, compute the percentage score, and get top 10
    return genre_counts.withColumn("rank", F.row_number().over(rank_window)) \
        .filter(F.col("rank") == 1) \
        .withColumn("genre_loyalty_score", F.round((F.col("count") / F.col("total_plays")) * 100, 2)) \
        .orderBy(F.col("genre_loyalty_score").desc()) \
        .select("user_id", F.col("genre").alias("top_genre"), "genre_loyalty_score") \
        .limit(10)


def get_night_owl_users(joined_df: DataFrame) -> DataFrame:
    """
    Task 4: Night Owl Users (12 AM - 5 AM)
    Identifies users who listen to music between 12 AM and 5 AM.
    """
    # Extract hour from string timestamp and filter between 0 (12 AM) and 4 (4:59 AM)
    return joined_df.withColumn("hour", F.hour(F.col("timestamp"))) \
        .filter((F.col("hour") >= 0) & (F.col("hour") < 5)) \
        .select("user_id") \
        .distinct()