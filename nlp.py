from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    'This is the first document',
    'This is the second document',
    'And this is the third one',
    'Is this the first document?'
]

#zapisanie zdan jako wektorow, usuniecie znakow specjalnych itp
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(corpus)


words = vectorizer.get_feature_names()

print(words)

#podobienstwo zdan w korpusie miedzy soba nawzajem

similarity_matrix = cosine_similarity(vectors)

print(similarity_matrix)