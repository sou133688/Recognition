import dataset
df=dataset.load_data()

import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
for i, col in enumerate(df.columns):
    plt.subplot(2,7,i+1)
    plt.hist(df[col])
    plt.title(col)
plt.tight_layout()
plt.show
