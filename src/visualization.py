import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

#phase 1:
def plot_class_distribution(df):
  df["label"].value_counts().plot(kind="bar")
  plt.title("Human: 0 vs AI: 1")
  plt.xlabel("label")
  plt.ylabel("count")
  plt.show()

#phase 2:
def plot_statistics(df):
  summary[["count_words", "average_sentence_length"]].plot(kind="bar", color=["skyblue","red"])
  plt.title("Human vs AI: text statistics")
  plt.xticks(rotation=0)
  plt.ylabel("average")
  plt.show()

def plot_length_distribution(df):
  df[df["label"]==0]["count_words"].plot(kind="hist",bins=50, alpha=0.7, label="Human", color="skyblue", density=True)
  df[df["label"]==1]["count_words"].plot(kind="hist",bins=50, alpha=0.4, label="AI", color="red", density=True)
  plt.title("Text length distribution")
  plt.xlabel("word count")
  plt.legend()
  plt.show()
