## iconv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.rand(10, 6)

sns.heatmap(data, cmap='coolwarm', xticklabels=False, yticklabels=False, linewidths=1, linecolor='black')
plt.savefig('iconv_heatmap.pdf')
plt.close()
