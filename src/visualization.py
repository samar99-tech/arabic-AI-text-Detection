import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display

#phase 1: class destribution
def plot_class_distribution(df):
  df["label"].value_counts().plot(kind="bar")
  plt.title("Human: 0 vs AI: 1")
  plt.xlabel("label")
  plt.ylabel("count")
  plt.show()

#phase 2: text statistcs
def plot_statistics(summary):
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

#word clouds:
reshaper = arabic_reshaper.ArabicReshaper(configuration={"support_ligatures": False})

def make_cloud(texts, title, font_path="Amiri-Regular.tff"):
  words = " ".join(texts).split()
  freq = Counter(words)

  freq_display= {
     get_display(reshaper.reshape(w))[::-1]: c
     for w, c in freq.items()
      }

  wc = WordCloud(font_path=font_path, width=1200, height=700, margin=15,
                 max_font_size=90, min_font_size=10, collocations= False,
                 background_color="white").generate_from_frequencies(freq_display)

  plt.figure(figsize=(14,8))
  plt.imshow(wc, interpolation="bilinear")
  plt.axis("off")
  plt.title(title)
  plt.savefig(
      title.replace(" ","_") + ".png",
      dpi=150, bbox_inches="tight", pad_inches=0.2)
  plt.show()


