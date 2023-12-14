import pandas as pd
import numpy as np
import pyodbc

import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import altair_viewer

from scipy import sparse
from scipy.sparse.csgraph import connected_components
import warnings

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

import string
import re

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('stopwords')


conn = pyodbc.connect('driver={SQL Server}; Server=bcompany-sqlserverprod.gojur.com.br,2788; UID=devconsulta; PWD=Dvc1142!!; Database=GPJWEB_1')
cursor = conn.cursor()

input_query = '''SELECT TOP 5000 * FROM JUR_PublicacaoArquivoItem where convert(date, dta_Publicacao) between '2023-08-01' and '2023-08-10';'''

df = pd.read_sql(input_query, conn)
df = df['des_Publicacao']
print(df.head(5))


df1 = df.apply(lambda text: str(text).lower())
df1 = df.apply(lambda text: re.sub('[0-9]+', ' ', text))
df1 = df.apply(lambda text: re.sub(r"\[(.*?)\]", " ", text))
df1 = df.apply(lambda text: re.sub(r"\s+", " ", text))
df1 = df.apply(lambda text: ' '.join([t for t in word_tokenize(text) if t not in stopwords.words('portuguese') and len(t) > 1]))
df1 = df.apply(lambda text: "".join([char for char in text if char not in string.punctuation]))


vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.05,
                             stop_words=stopwords.words('portuguese'))
vecs = vectorizer.fit_transform(df1)


scaller = StandardScaler(with_mean=False)
scalled_data = scaller.fit_transform(vecs)


pca = PCA(n_components=2)
data = pca.fit_transform(scalled_data.toarray())
X = data[:, 0]
y = data[:, 1]


def fix_connectivity(data, connectivity, affinity):

    n_samples = data.shape[0]
    if (connectivity.shape[0] != n_samples or
        connectivity.shape[1] != n_samples):
        raise ValueError('Wrong shape for connectivity matrix: %s'
                         'when x is %s' % (connectivity.shape, data.shape))
    

    connectivity = connectivity + connectivity.T


    if not sparse.isspmatrix_lil(connectivity):
        if not sparse.isspmatrix(connectivity):
            connectivity = sparse.lil_matrix(connectivity)
        else:
            connectivity = connectivity.tolil()


    n_connected_components, labels = connected_components(connectivity)


    if n_connected_components > 1:
        warnings.warn('O numero de componentes conectados da matriz e %d > 1.'
                      'Complete para evitar parar a tree muito cedo.' 
                      % n_connected_components, stacklevel=2)
        
        for i in range(n_connected_components):
            idx_i = np.where(labels == i)[0]
            xi = data[idx_i]
            for j in range(i):
                idx_j = np.where(labels == j)[0]
                xj = data[idx_j]
                d = pairwise_distances(xi, xj, metric=affinity)
                ii, jj = np.where(d == np.min(d))
                ii = ii[0]
                jj = jj[0]
                connectivity[idx_i[ii], idx_j[jj]] == True
                connectivity[idx_j[jj], idx_i[ii]] == True


    return connectivity, n_connected_components


def get_distances(data, model, mode='l2'):
    distances = []
    weights = []
    children = model.children_
    dims = (data.shape[1], 1)
    dist_cache = {}
    weight_cache = {}

    for childs in children:
        c1 = data[childs[0]].reshape(dims)
        c2 = data[childs[1]].reshape(dims)
        c1_dist = 0
        c1_weight = 1
        c2_dist = 0
        c2_weight = 1

        if childs [0] in dist_cache.keys():
            c1_dist =   dist_cache[childs[0]]
            c1_weight = weight_cache[childs[0]]
        if childs[1] in dist_cache.keys():
            c2_dist = dist_cache[childs[1]]
            c2_weight = weight_cache[childs[1]]
        d = np.linalg.norm(c1 - c2)
        cc = ((c1_weight * c1) + (c2_weight * c2))/(c1_weight + c2_weight)

        x = np.vstack((x, cc.T))

        newchild_id = x.shape[0]-1

        if mode=='l2':
            added_dist = (c1_dist**2+c2_dist**2)**0.5
            d_new = (d**2+added_dist**2)**0.5
        elif mode=='max':
            d_new = max(d, c1_dist, c2_dist)
        else:
            d_new = d

        w_new = (c1_weight + c2_weight)
        dist_cache[newchild_id] = d_new
        weight_cache[newchild_id] = w_new

        distances.append(d_new)
        weights.append(w_new)
    

    return distances, weights


for knn in data:
    knn = NearestNeighbors(n_neighbors=100, metric='euclidean').fit(vecs)


k= 8
model = AgglomerativeClustering(n_clusters=k, metric='euclidean',
                                linkage='ward', connectivity=knn.kneighbors_graph(vecs),
                                compute_full_tree=True)
model.fit(data)
label = model.labels_

plt.figure(figsize=(14, 8))
plt.title('Agglomerative Clustering')
plt.xlabel('X', fontdict={'fontsize' : 16})
plt.ylabel('Y', fontdict={'fontsize' : 16})
sns.scatterplot(x= X, y= y, hue=label, palette='tab10')


words = vectorizer.get_feature_names_out()

def get_cluster_keywords(vecs, words, label):
    cluster_keyword_id = {cluster_id: {} for cluster_id in set(label)}

    for vec, cluster_id in zip(vecs, label):

        for j in vec.nonzero()[1]:

            if j not in cluster_keyword_id[cluster_id]:
                cluster_keyword_id[cluster_id][j] = 0

                cluster_keyword_id[cluster_id][j] += 1

    return {
        cluster_id: [
            words[keyword_id]

            for keyword_id, count in sorted(
                keyword_id_count.items(),
                key=lambda x : x[1],
                reverse=True
            )
        ] for cluster_id, keyword_id_count in cluster_keyword_id.items()
    }

palavras_freq = get_cluster_keywords(vecs, words, label)


text_counts = {}
for cluster_id in range(k):
    text_counts[cluster_id] = len(df[label == cluster_id])

df_new = pd.read_sql(input_query, conn)
df_new = df_new['cod_PublicacaoArquivoItem']

def get_cluster_text(df, label):

    cluster_texts = []

    df = pd.concat([df, df_new], axis=1)
    for cluster_id in range(k):
        index = np.where(label == cluster_id)[0]
        cluster_text = df.loc[index]
        for text in cluster_text['cod_PublicacaoArquivoItem']:
            current_text = text
            cluster_texts.append({
                "cluster_id": cluster_id,
                "text": current_text
            })

    return cluster_texts

cluster_texts = get_cluster_text(df, label)

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)
pd.set_option('display.width', 100000)
np.set_printoptions(threshold=np.inf, linewidth=np.nan)

df_texto = pd.DataFrame(cluster_texts)
df_texto.to_csv('resultados_cluster2.csv', index=False,
                sep=',')


clusters = pd.DataFrame({
    "labels": label,
    "df": df,
    "X": X,
    "y": y
})


alt.renderers.enable('default')

chart = alt.Chart(clusters).mark_circle(size=100).encode(
    x='X',
    y='y',
    color= 'labels:N',
    tooltip='df').properties(
        width=700, 
        height=500).interactive()

altair_viewer.show(chart)


conn.close()