# testing_metrics_plot.py
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from ssm_utils import *
from ssm_config import *
import pandas as pd


bone = 'left_femur'

df = pd.read_csv(os.path.join(OUT_DIR, bone,'eval_metric_resampled.csv'))
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
sns.stripplot(x='INPUT TYPE', y='DSC', data=df, hue='FILENAME', palette='Set3')
plt.ylim(0, 1)
plt.title('Dice Similarity Coefficient')

plt.subplot(1, 4, 2)
sns.stripplot(x='INPUT TYPE', y='HDmax', data=df, hue='FILENAME', palette='Set3', legend=False)
plt.title('Max Hausdorff Distance')

plt.subplot(1, 4, 3)
sns.stripplot(x='INPUT TYPE', y='HD95', data=df, hue='FILENAME', palette='Set3', legend=False)
plt.title('95% Hausdorff Distance')

plt.subplot(1, 4, 4)
sns.stripplot(x='INPUT TYPE', y='HDrmse', data=df, hue='FILENAME', palette='Set3', legend=False)
plt.title('RMSE of Hausdorff Distance')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, bone, 'eval_metric_resampled.png'))
plt.show()
plt.close()


# box plot
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
sns.boxplot(x='INPUT TYPE', y='DSC', data=df, palette='Set3')
plt.title('Dice Similarity Coefficient')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.subplot(1, 4, 2)
sns.boxplot(x='INPUT TYPE', y='HDmax', data=df, palette='Set3')
plt.title('Max Hausdorff Distance')

plt.subplot(1, 4, 3)
sns.boxplot(x='INPUT TYPE', y='HD95', data=df, palette='Set3')
plt.title('95% Hausdorff Distance')

plt.subplot(1, 4, 4)
sns.boxplot(x='INPUT TYPE', y='HDrmse', data=df, palette='Set3')
plt.title('RMSE of Hausdorff Distance')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, bone, 'eval_metric_resampled_boxplot.png'))
plt.show()
plt.close()


grouped = df.groupby('INPUT TYPE')[['DSC', 'HDmax', 'HD95', 'HDrmse']].agg(['mean', 'std'])
print(grouped)
grouped.to_csv(os.path.join(OUT_DIR, bone, 'eval_metric_resampled_grouped.csv'))