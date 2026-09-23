import matplotlib as plt
def plot_class_distribution(df):
  df["label"].value_counts().plot(kind="bar")
  plt.title("Human: 0 vs AI: 1")
  plt.xlabel("label")
  plt.ylabel("count")
  plt.show()
