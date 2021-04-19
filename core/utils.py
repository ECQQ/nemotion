import numpy as np
import pandas as pd

def get_closest_vector(pivots, names, emo_vector, embedding):
	words = []
	classes = []
	for w in emo_vector:
	    v = embedding[w]
	    mindist = 99999
	    label = ''
	    for p, n in zip(pivots, names):
	        dist = np.linalg.norm(v-p)
	        if dist < mindist:
	            mindist = dist
	            label = n
	    words.append(w)
	    classes.append(label)

	df = pd.DataFrame()
	df['word'] = words
	df['label'] = classes

	return df
