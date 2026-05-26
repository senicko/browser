# Dataset

```
=============================================
CORPUS TEXT METRICS SUMMARY
=============================================
Total Articles:		        161,667
Average Words/Article:	         234.12
Max Words (Longest):	            451
Min Words (Shortest):	             20
=============================================
```

# Model

Moduł przygotowujący modele pozwalające przeszukiwać korpus tekstu. Moduł implementuje podejścia:

- Bag of Words
- LSA
- BM25

## Bag of Words (BoW) oparte o TD-IDF

Model jest reprezentowany przez macierz `m x n`, gdzie `m` to liczba dokumentów, a `n` to liczba słów w wyznaczonym słowniku. Słowa w słowniku:

1. Muszą być w conajmniej 5 artykułach (`min_df=5`)
2. Muszą być w co najwyżej 80% wszystkich artykułów (`max_df=0.8`)
 
Wartości w komurkach macierzy są obliczane w oparciu o tf-idf -- reprezentują ważność słowa. Wykorzystana została implementacja tf-idf `sklearn.feature_extraction.text.TfidfVectorizer`.

Przeszukiwanie polega na porównywaniu wektorowej reprezentacji `1 x n` zapytania z wektorowymi reprezentacjiami `1 x n` kolejnych dokumentów na podstawie podobieństwa cosinusowego.

## Latent Semantic Analysis (LSA)

Podejście `LSA` polega na przeprowadzeniu dekompozycji `SVD` modelu `BoW`, oraz przybliżeniu go wybraną liczbą wartości własnych. Ma to na celu uogólnić znaczenie dokumentów, co obrazowo "powoduje przeszukiwanie wśród konceptów, a nie poszczególnych słów".

Do obliczenia modelu `LSA` został stworzony pipeline, który przekształca otrzymany wcześniej model `BoW`:

1. `sklearn.decomposition.TruncatedSVD` -- przybliża macierz n wartościami własnymi (wykorzystując algorytm przybliżania SVD `sklearn.utils.extmath.randomized_svd`, co jest wydajniejsze obliczeniowo)
2. `sklearn.preprocessing.Normalizer` -- odpowiada za normalizację


## BM25(+)

Model wykorzystuje funkcję rankingową BM25. W ramach projektu zaimplementowałem Transformer i Vectorizer dla funkcji rankingowej BM25 (bm25.py). Macierzą wynikową jest macierz z obliczoną punktacją dla każdego słowa ze słownika dla każdego dokumentu. Wyszukiwanie odbywa się poprzez zamianę zapytania na wektor 1 hot encoding, który następnie jest mnożony z każdym dokumentem w celu obliczenia finalnego wyniku.
