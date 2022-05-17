import pandas as pd
import numpy as np
from classifier import Classifier

c = Classifier("custom_word_2_vec")

# Load Data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Fit KNN (word2vec already fit)
c.fit(np.array(df_train["title"].astype(str))[:10000], np.array(df_train["categories"].astype(str))[:10000])

# Simple exampe
print("Ground Truth", np.array(df_test["categories"])[:10])
print("Predicted", c.predict(np.array(df_test["title"]))[:10])