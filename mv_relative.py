#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

rating_f = 'ml-latest-small/ratings.csv'
link_f	 = 'ml-latest-small/links.csv'

df = pd.read_csv(rating_f,sep=',')
df_id = pd.read_csv(link_f,sep=',')
df = pd.merge(df,df_id,on=['movieId'])

rating_matrix = np.zeros((df.userId.unique().shape[0],max(df.movieId)))
for row in df.itertuples():
	rating_matrix[row[1]-1,row[2]-1] = row[3]
rating_matrix = rating_matrix[:,:9000]

#Evaluate sparsity of matrix
sparsity = float(len(rating_matrix.nonzero()[0]))
sparsity /= (rating_matrix.shape[0]*rating_matrix.shape[1])
sparsity *= 100
print('Sparsity is {0}%'.format(sparsity))

#Splite to train/test matrix
train_matrix = rating_matrix.copy()
test_matrix = np.zeros(rating_matrix.shape)

for i in range(rating_matrix.shape[0]):
	rating_index = np.random.choice(rating_matrix[i,:].nonzero()[0],size=10,replace=True)
	train_matrix[i,rating_index] = 0.0
	test_matrix[i,rating_index] = rating_matrix[i,rating_index]

#Cosine similarity
similarity_usr = train_matrix.dot(train_matrix.T) + 1e-9
norms = np.array([np.sqrt(np.diagonal(similarity_usr))])
similarity_usr = (similarity_usr/(norms*norms.T))

similarity_mv = train_matrix.T.dot(train_matrix) + 1e-9
norms = np.array([np.sqrt(np.diagonal(similarity_mv))])
similarity_mv = (similarity_mv/(norms*norms.T))

prediction = similarity_usr.dot(train_matrix)/np.array([np.abs(similarity_usr).sum(axis=1)]).T
prediction = prediction[test_matrix.nonzero()]
test_vector = test_matrix[test_matrix.nonzero()]
mse = mean_squared_error(prediction,test_vector)

print('mse: {0}'.format(mse))

#Test
import requests
import json
from IPython.display import Image
from IPython.display import display
from IPython.display import HTML

k = 10
n_display = 5
base_mv_idx = 0
idx_to_mv = {}

for row in df_id.itertuples():
	idx_to_mv[row[1]-1] = row[2]
mv = [idx_to_mv[x] for x in np.argsort(similarity_mv[base_mv_idx])[:-k-1:-1]]
mv = filter(lambda imdb: len(str(imdb))==6, mv)
mv = list(mv)[:n_display]

#Get posters from Movie Database by API
headers = {'Accept':'application/json'}
payload = {'api_key':'20047cd838219fb54d1f8fc32c45cda4'}
response = requests.get('http://api.themoviedb.org/3/configuration',params=payload,headers=headers)
response = json.loads(response.text)

base_url = response['images']['base_url']+'w185'

def get_poster(imdb,base_url):
	#query themovie.org API for movie poster path.
	imdb_id = 'tt0{0}'.format(imdb)
	movie_url = 'http://api.themoviedb.org/3/movie/{:}/images'.format(imdb_id)
	response = requests.get(movie_url,params=payload,headers=headers)

	try:
		file_path = json.loads(response.text)['posters'][0]['file_path']
	except:
		print('Something wrong, cannot get the poster for imdb id: {0}!'.format(imdb))

	return base_url+file_path


URL = [0]*len(mv)
for i,m in enumerate(mv):
	URL[i] = get_poster(m,base_url)

images = ''
for i in range(len(mv)):
	images+="<img style='width: 100px; margin: 0px; float: left; border: 1px solid black;' src='%s' />" % URL[i]

#print(images)
display(HTML(images))
