import pandas as pd
from matplotlib import pyplot as plt

#read in csv files:

synth_stats = pd.read_csv('synth_stats.csv')
real_stats = pd.read_csv('real_stats.csv')

print(real_stats.to_latex(index=False))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
# synth_stats.plot(kind='box', color='blue', ax=ax1,showfliers = False)
# real_stats.plot(kind='box', color='red', ax=ax2,showfliers = False)

# # set the title and axis labels
# ax1.set_title('CIGAN Synthetic Samples')
# ax1.set_ylabel('Value')
# ax1.set_xlabel('Variable Name')

# ax2.set_title('Sachs Real Samples')
# ax2.set_ylabel('Value')
# ax2.set_xlabel('Variable Name')

# ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
# ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)



# #set the plot title:
# plt.suptitle('Summary Statistics for CIGAN Synthetic Samples vs Sachs Real Samples')

# plt.show()