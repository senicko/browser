# Dataset

Zbiór danych to około 25 tysięcy artykułów z angielskiej Wikipedii kategorii `Culture` (do 3 pod-kategorii w dół). Tytuły artykułów zostały pobrane przy pomocy narzędzia [PetScan](https://petscan.wmcloud.org/), a następnie pobrane przygotowanym skryptem odpytującym API Wikipedii. Nie jest to z definicji web-scraper, ale projekt udostępnia skrypt pozwalający pobrać dowolne artykuły z Wikipedii przy pomocy publicznego API.

Ponieważ niektóre artykuły są bardzo długie, zostały podzielone na mniejsze fragmenty przy pomocy `langchain_text_splitters.RecursiveCharacterTextSplitter`. Dzięki temu korpus tekstu składa się z 161,667 dokumentów podobnych wielkości, co poprawia jakość wyszukiwania.

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

## Przygotowanie korpusu (`scraper.py`)

Pipeline budowania korpusu (`build_corpus`) działa następująco:

|  Etap | Opis |
|------|------|
| Źródło tytułów | Plik CSV (`culture_petscan.csv`) wczytywany przez `load_titles`; z każdego wiersza brany jest tytuł (kolumna 1). |
| Pobieranie treści | `fetch_wikipedia_articles` odpytuje `https://en.wikipedia.org/w/api.php`. Wyniki mogą być cache’owane w `artifacts/culture_articles_cache.json`. Przy kodach 429/502/503/504 stosowany jest backoff wg nagłówka `Retry-After`. |
| Czyszczenie tekstu | `_clean_article_text` usuwa nagłówki w stylu wikitext (`== Sekcja ==`) oraz obcina artykuł przed sekcjami typu References, External links, See also itd. |
| Chunking | `RecursiveCharacterTextSplitter`: `chunk_size=300` tokenów, `chunk_overlap=100` tokenów, `length_function=StemTokenizer.token_count` (liczba tokenów po stemowaniu i usunięciu stop-słów). |
| Filtrowanie | Fragmenty krótsze niż `MIN_CHUNK_TOKENS` (20 tokenów) są odrzucane. |
| Zapis | Lista obiektów `Document` (tytuł, tekst fragmentu, URL Wikipedii) serializowana do `artifacts/corpus.pkl` przez `joblib`. |

Każdy element korpusu to **fragment artykułu** (chunk), nie cały artykuł. Ten sam tytuł może wystąpić w wielu dokumentach.

## Tokenizacja (`tokenizer.py`)

Wszystkie modele wyszukiwania używają wspólnego `StemTokenizer`:

- normalizacja białych znaków i podział przez `nltk.tokenize.word_tokenize`,
- filtrowanie: tylko tokeny alfabetyczne, bez angielskich stop-słów (`nltk.corpus.stopwords`),
- stemowanie: `PorterStemmer`.

Ten sam tokenizer jest używany przy budowie indeksów i przy zapytaniach użytkownika, co zapewnia spójność słownika.

---

# Model

Moduł `packages/model` buduje artefakty wyszukiwania (`artifacts/*.pkl`) i udostępnia trzy klasy zapytań w `models.py`: `BoWModel`, `LSAModel`, `BM25Model`. Wspólny kontrakt to `Model.query(phrase, top_n=10) -> list[ModelResponse]`, gdzie `ModelResponse` zawiera `title`, `url`, `score`, `text`.

Artefakty ładowane są leniwie (`functools.lru_cache`) przy pierwszym zapytaniu.

## Budowa indeksów (`main.py`)

Skrypt `python -m model.main`:

1. Ładuje korpus z `corpus.pkl` lub buduje go przez `build_corpus`, jeśli plik nie istnieje.
2. Pobiera listę tekstów fragmentów: `corpus_texts`.
3. Trenuje i zapisuje trzy reprezentacje (BoW/LSA/BM25) z tymi samymi filtrami słownika: `min_df=5`, `max_df=0.8`.

### Rozmiar słownika (wytrenowane artefakty)

Po załadowaniu modeli z `artifacts/` (np. `len(vectorizer.vocabulary_)`):

| Parametr | Wartość |
|----------|---------|
| Rozmiar słownika (BoW / BM25) | **105,047** terminów |
| Liczba dokumentów (`m`) | 161,667 |
| Wymiar macierzy `bow.pkl` | 161,667 × 105,047 |
| Wymiar macierzy `lsa_bow.pkl` | 161,667 × 200 |

BoW (`TfidfVectorizer`) i BM25 (`BM25Vectorizer`) mają ten sam rozmiar słownika — oba używają `StemTokenizer` oraz `min_df=5`, `max_df=0.8` na tym samym korpusie. LSA redukuje reprezentację do 200 składowych (`n_components = min(200, m - 1, n - 1)`).

| Plik | Zawartość |
|------|-----------|
| `bow.pkl` | Macierz rzadka TF-IDF dokumentów (`m × n`) |
| `vectorizer.pkl` | `TfidfVectorizer` dopasowany do korpusu |
| `lsa_bow.pkl` | Macierz dokumentów w przestrzeni LSA |
| `lsa_vectorizer.pkl` | Pipeline: TF-IDF → SVD → normalizacja |
| `bm25.pkl` | Macierz rzadka wag BM25 dokumentów |
| `bm25_vectorizer.pkl` | `BM25Vectorizer` (słownik + parametry BM25) |

---

## Bag of Words (BoW) — TF-IDF

**Trenowanie:** `TfidfVectorizer(tokenizer=StemTokenizer(), min_df=5, max_df=0.8)` → `fit_transform(corpus_texts)` → macierz `bow` o wymiarach `m × n` (`m` — liczba chunków, `n` — rozmiar słownika). Na obecnym korpusie: **161,667 × 105,047**.

Filtry słownika:

- słowo musi wystąpić w co najmniej 5 dokumentach (`min_df=5`),
- słowo nie może występować w więcej niż 80% dokumentów (`max_df=0.8`).

**Zapytanie (`BoWModel.query`):**

1. `phrase_vec = vectorizer.transform([phrase])` — wektor zapytania `1 × n` w przestrzeni TF-IDF.
2. `cosine_similarity(phrase_vec, bow)` — podobieństwo cosinusowe zapytania do każdego dokumentu.
3. `argsort` malejąco, `top_n` indeksów z najwyższym wynikiem.
4. Mapowanie indeksów na `ModelResponse` z korpusu.

Wynik `score` to wartość podobieństwa cosinusowego z zakresu typowo `[0, 1]`.

---

## Latent Semantic Analysis (LSA)

**Trenowanie:** na macierzy TF-IDF (`bow`) stosowany jest pipeline:

1. `TruncatedSVD(n_components, random_state=42)` — liczba składowych:
   ```text
   n_components = min(200, m - 1, n - 1)
   ```
   (algorytm `randomized_svd` w sklearn).
2. `Normalizer(copy=False)` — normalizacja wierszy (L2).

Wynik `lsa_bow` to dokumenty w przestrzeni o wymiarze `n_components` (maks. 200).

**Wektorzator zapytań:** `Pipeline([("tfidf", tfidf_vectorizer), ("svd", lsa_pipeline)])` — ten sam TF-IDF co przy BoW, potem ta sama redukcja SVD i normalizacja.

**Zapytanie (`LSAModel.query`):** identyczny schemat co BoW, ale w przestrzeni LSA (`load_lsa_vectorizer`, `load_lsa_bow`) i z `cosine_similarity`.

Semantycznie: LSA grupuje współwystępowania terminów; wyszukiwanie odbywa się w przestrzeni „tematów” (składowych SVD), a nie surowych tokenów.

---

## BM25(+)

Implementacja w `bm25.py` (wzorowana na [BM25-scikit-learn](https://github.com/nocchi1/BM25-scikit-learn/blob/main/bm25.py)):

- **`BM25Vectorizer`** — rozszerza `CountVectorizer`; przy `fit`/`fit_transform` buduje macierz częstości, następnie **`BM25Transformer`** zamienia ją na wagi BM25.
- **`BM25Transformer.transform`** — dla każdej niezerowej komórki stosuje wzór BM25 z parametrami `k1`, `b`, opcjonalnie `delta` (BM25+), oraz mnożenie przez `idf_` wyliczone w `fit`.

**Parametry w projekcie** (`main.py`):

| Parametr | Wartość | Znaczenie |
|----------|---------|-----------|
| `k1` | `2.0` | Nasycenie częstości terminu w dokumencie |
| `b` | `0.4` | Wpływ długości dokumentu (mniejszy niż typowe 0.75 — mniejsza kara za krótkie chunki) |
| `delta` | `1` | Wariant **BM25+** (stały dodatek do wagi) |
| `min_df` / `max_df` | `5` / `0.8` | Jak w TF-IDF |
| `norm` | `None` | Bez dodatkowej normalizacji L1/L2 po BM25 |

**Trenowanie:** `bm25 = bm25_vectorizer.fit_transform(corpus_texts)` — macierz rzadka `m × n` z wagami BM25 per (dokument, termin). Na obecnym korpusie: **161,667 × 105,047** (ten sam słownik co BoW).

**Zapytanie (`BM25Model.query`):**

1. Zapytanie wektoryzowane przez **warstwę liczenia** vectorizera, bez ponownego `BM25Transformer`:
   `CountVectorizer.transform(bm25_vectorizer, [phrase])`.
2. Jeśli zapytanie nie zawiera żadnego terminu ze słownika (`phrase_vec.nnz == 0`), zwracana jest pusta lista.
3. Wszystkie niezerowe wagi zapytania ustawiane są na `1` (`phrase_vec.data.fill(1)`) — zapytanie traktowane jak **binarne** (obecność terminu, bez TF zapytania).
4. Wyniki: `scores = bm25_matrix @ phrase_vec.T` (iloczyn macierzowy — suma wag BM25 dokumentów dla terminów z zapytania).
5. Zwracane są `top_n` indeksów z `scores > 0`.

Wynik `score` to surowa suma wag BM25, nie podobieństwo cosinusowe (inna skala niż BoW/LSA).

---

# Client

Aplikacja Gradio (`packages/client/src/client/main.py`) uruchamia równoległe wyszukiwanie **trzema modelami** dla jednego zapytania.

## Prezentacja wyników

| Model | Format `score` w UI | Kolor nagłówka |
|-------|---------------------|----------------|
| BoW, LSA | `score × 100` z jednym miejscem po przecinku + `%` | `get_color_for_score(score)` — odcień zieleni (HSL, `hue = score × 120`) |
| BM25 | wartość z 4 miejscami po przecinku | bez kolorowania (`inherit`) |

---

# Architektura pakietów

```
browser/
├── packages/
│   ├── schema/     # Document(title, text, url)
│   ├── model/      # scraper, tokenizer, bm25, main (trening), models (zapytania)
│   └── client/     # Gradio UI
└── packages/model/artifacts/   # corpus.pkl, bow.pkl, …
```

**Uruchomienie (po zbudowaniu artefaktów):** trening indeksów — `python -m model.main`; UI — entry point pakietu `client` (Gradio).
