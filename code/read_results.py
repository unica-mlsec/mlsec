import os

import pandas as pd

list_results = os.listdir("results")
for f in list_results:
	df = pd.read_csv(os.path.join("results", f))

	print(f"model: {f[:-len('.csv')]}")
	accuracy = (df['label'] == df['pred']).mean()
	print("accuracy: ", accuracy)

	robust_accuracy = (df['label'] == df['adv']).mean()
	print("robust accuracy:", robust_accuracy)