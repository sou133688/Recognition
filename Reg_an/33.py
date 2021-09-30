import dataset
df = dataset.load_data()
df_corr = df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
sns.heatmap(df_corr, annot=True)
plt.title("Heat map")
plt.show()