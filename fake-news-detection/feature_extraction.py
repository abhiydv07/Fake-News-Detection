"""
Feature Extraction Module
Handles converting text data to numerical features using TF-IDF, Count Vectorizer,
metadata features, sentiment analysis, and Word2Vec embeddings.
"""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

# Sentiment lexicon (simple approach - no NLTK VADER dependency)
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant',
    'outstanding', 'superb', 'perfect', 'best', 'better', 'success', 'successful',
    'achieve', 'achieved', 'breakthrough', 'innovative', 'improve', 'improved',
    'progress', 'advance', 'advanced', 'develop', 'developed', 'discovery',
    'discover', 'discoveries', 'positive', 'promising', 'effective', 'benefit',
    'benefits', 'profit', 'profitable', 'growth', 'growing', 'increase', 'increased',
    'record', 'high', 'approval', 'approved', 'launch', 'launched', 'celebrate',
    'victory', 'win', 'won', 'triumph', 'exceed', 'exceeded', 'remarkable',
    'exceptional', 'impressive', 'incredible', 'magnificent', 'revolutionary',
    'approved', 'endorsed', 'grant', 'granted', 'funding', 'funded', 'support',
    'supported', 'rescue', 'rescued', 'safe', 'safety', 'secure', 'security',
    'healthy', 'health', 'heal', 'healed', 'cure', 'cured', 'treatment',
    'treated', 'recover', 'recovered', 'recovery', 'solve', 'solved', 'solution',
    'clean', 'renewable', 'sustainable', 'protect', 'protected', 'conservation',
    'celebrate', 'celebration', 'honor', 'honest', 'trust', 'trusted', 'truth'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'horrible', 'awful', 'disgusting', 'worst', 'worse',
    'fail', 'failed', 'failure', 'disaster', 'disastrous', 'catastrophe',
    'dangerous', 'harmful', 'toxic', 'poison', 'poisonous', 'deadly', 'death',
    'kill', 'killed', 'murder', 'crime', 'criminal', 'fraud', 'fraudulent',
    'fake', 'hoax', 'scam', 'deceive', 'deceived', 'deception', 'lie', 'liar',
    'lying', 'false', 'fabricate', 'fabricated', 'conspiracy', 'conspiracy',
    'coverup', 'cover-up', 'suppress', 'suppressed', 'hiding', 'hidden',
    'secret', 'secretly', 'exposed', 'expose', 'scandal', 'corruption',
    'corrupt', 'bribe', 'bribed', 'bribery', 'threat', 'threaten', 'threatened',
    'danger', 'risk', 'risky', 'warning', 'warn', 'warned', 'panic', 'panic',
    'fear', 'afraid', 'terrified', 'horror', 'shocking', 'shock', 'alarming',
    'alarm', 'crisis', 'emergency', 'disaster', 'devastating', 'devastation',
    'ruin', 'ruined', 'destroy', 'destroyed', 'destruction', 'damage', 'damaged',
    'hurt', 'injure', 'injured', 'injury', 'suffer', 'suffering', 'pain',
    'agony', 'torture', 'abuse', 'exploit', 'exploited', 'manipulate',
    'manipulated', 'brainwash', 'brainwash', 'zombie', 'control', 'controlled'
}

CONSPIRACY_WORDS = {
    'conspiracy', 'conspirator', 'secret', 'secretly', 'hidden', 'coverup',
    'cover-up', 'suppressed', 'government', 'government', 'aliens', 'alien',
    'ufos', 'ufo', 'hollow', 'flat earth', 'chemtrails', 'chemtrail',
    'microchip', 'mind control', 'shapeshifter', 'reptile', ' Illuminati',
    'new world order', 'deep state', 'shadow', 'underground', 'bunker',
    'hoax', 'fabricated', 'fake', 'deception', 'deceive', 'lie', 'liar',
    'annunaki', 'nibiru', 'apocalypse', 'doomsday', 'end of the world',
    'zombie', 'undead', 'reanimated', 'clone', 'cloning', 'simulation',
    'simulation theory', 'glitch in the matrix', 'moon landing fake',
    'moon is fake', 'hollow moon', 'moon is hollow', 'cheese', 'astronaut',
    'fake doctor', 'fake scientist', 'miracle cure', 'miracle pill',
    'instant cure', 'cures all', 'cure all', 'snake oil', 'pseudoscience',
    'pseudoscientific', 'magnetic water', 'urine therapy', 'bleach cure',
    'bleach drink', 'drinking bleach', 'drinking urine', 'raw food cure',
    'bananas cure', 'cancer cure', 'cures cancer', 'cures autism'
}

SCIENCE_WORDS = {
    'research', 'researchers', 'study', 'studies', 'experiment', 'experiment',
    'laboratory', 'lab', 'clinical', 'clinical trial', 'peer-reviewed',
    'peer reviewed', 'published', 'journal', 'scientific', 'science',
    'scientist', 'scientists', 'professor', 'university', 'college',
    'institute', 'instituted', 'discovery', 'discovered', 'breakthrough',
    'innovation', 'innovative', 'technology', 'technological', 'engineering',
    'engineer', 'algorithm', 'computing', 'quantum', 'genome', 'genomic',
    'genetic', 'gene', 'dna', 'rna', 'protein', 'molecular', 'cellular',
    'evolution', 'evolutionary', 'ecosystem', 'ecosystems', 'biodiversity',
    'astronomy', 'astronomer', 'telescope', 'spacecraft', 'satellite',
    'mission', 'orbit', 'launch', 'rocket', 'propulsion', 'physics',
    'chemistry', 'biology', 'medicine', 'medical', 'therapy', 'treatment',
    'vaccine', 'vaccination', 'immunotherapy', 'immunization', 'crispr',
    'machine learning', 'artificial intelligence', 'ai', 'neural network',
    'deep learning', 'nlp', 'natural language processing', 'data science',
    'superconductor', 'semiconductor', 'battery', 'photovoltaic', 'solar',
    'wind', 'renewable', 'nuclear', 'fusion', 'fission', 'energy',
    'materials', 'metamaterials', 'nanotechnology', 'nanomaterial'
}


class MetadataExtractor:
    """Extract metadata features from text."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = []

    def _extract_single(self, text, raw_text=None):
        """Extract metadata features from a single text."""
        if raw_text is None:
            raw_text = text

        features = {}
        # Length features
        features['char_count'] = len(raw_text)
        features['word_count'] = len(raw_text.split())
        features['avg_word_length'] = np.mean([len(w) for w in raw_text.split()]) if raw_text.split() else 0
        features['sentence_count'] = max(1, raw_text.count('.') + raw_text.count('!') + raw_text.count('?'))

        # Punctuation features
        features['exclamation_count'] = raw_text.count('!')
        features['question_count'] = raw_text.count('?')
        features['comma_count'] = raw_text.count(',')
        features['semicolon_count'] = raw_text.count(';')
        features['colon_count'] = raw_text.count(':')
        features['quote_count'] = raw_text.count('"') + raw_text.count("'")
        features['ellipsis_count'] = raw_text.count('...')

        # Capital features
        words = raw_text.split()
        features['caps_ratio'] = sum(1 for w in words if w.isupper() and len(w) > 1) / max(1, len(words))
        features['all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)

        # Readability proxy
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        features['unique_word_ratio'] = len(set(raw_text.lower().split())) / max(1, len(words))

        # Special patterns
        features['has_numbers'] = int(bool(re.search(r'\d+', raw_text)))
        features['number_count'] = len(re.findall(r'\d+', raw_text))
        features['url_count'] = len(re.findall(r'http\S+|www\S+', raw_text))
        features['has_url'] = int(features['url_count'] > 0)

        # Sentiment word ratios
        lower_text = set(raw_text.lower().split())
        word_count = max(1, len(words))
        features['positive_ratio'] = len(lower_text & POSITIVE_WORDS) / word_count
        features['negative_ratio'] = len(lower_text & NEGATIVE_WORDS) / word_count
        features['conspiracy_ratio'] = len(lower_text & CONSPIRACY_WORDS) / word_count
        features['science_ratio'] = len(lower_text & SCIENCE_WORDS) / word_count
        features['sentiment_diff'] = features['positive_ratio'] - features['negative_ratio']

        # Suspicious patterns (common in fake news)
        features['has_excessive_caps'] = int(features['caps_ratio'] > 0.3)
        features['has_excessive_exclamations'] = int(features['exclamation_count'] > 3)
        features['question_mark_count'] = features['question_count']

        return features

    def fit_transform(self, texts, raw_texts=None):
        """Extract metadata features from a list of texts."""
        if raw_texts is None:
            raw_texts = texts

        all_features = []
        for i, text in enumerate(texts):
            raw = raw_texts[i] if isinstance(raw_texts, list) else raw_texts.iloc[i]
            features = self._extract_single(text, raw)
            all_features.append(features)

        df = pd.DataFrame(all_features)
        self.feature_names_ = list(df.columns)
        self.scaler.fit(df.values)
        scaled = self.scaler.transform(df.values)
        return csr_matrix(scaled)

    def transform(self, texts, raw_texts=None):
        """Transform new texts using fitted scaler."""
        if raw_texts is None:
            raw_texts = texts

        all_features = []
        for i, text in enumerate(texts):
            raw = raw_texts[i] if isinstance(raw_texts, list) else raw_texts.iloc[i]
            features = self._extract_single(text, raw)
            all_features.append(features)

        df = pd.DataFrame(all_features, columns=self.feature_names_)
        scaled = self.scaler.transform(df.values)
        return csr_matrix(scaled)


class SentimentExtractor:
    """Extract sentiment features from text."""

    def __init__(self):
        self.feature_names_ = []

    def _extract_single(self, text):
        """Extract sentiment features from a single text."""
        words = text.lower().split()
        word_count = max(1, len(words))
        word_set = set(words)

        pos_count = len(word_set & POSITIVE_WORDS)
        neg_count = len(word_set & NEGATIVE_WORDS)

        features = {}
        features['positive_word_count'] = pos_count
        features['negative_word_count'] = neg_count
        features['positive_ratio'] = pos_count / word_count
        features['negative_ratio'] = neg_count / word_count
        features['sentiment_polarity'] = (pos_count - neg_count) / max(1, pos_count + neg_count)
        features['sentiment_subjectivity'] = (pos_count + neg_count) / word_count

        # Emotional intensity
        intense_words = {'shocking', 'incredible', 'amazing', 'terrible', 'horrible',
                        'miracle', 'miraculous', 'unbelievable', 'outrageous', 'disgusting'}
        features['emotional_intensity'] = len(word_set & intense_words) / word_count

        return features

    def fit_transform(self, texts):
        """Extract sentiment features."""
        all_features = [self._extract_single(t) for t in texts]
        df = pd.DataFrame(all_features)
        self.feature_names_ = list(df.columns)
        return csr_matrix(df.values)

    def transform(self, texts):
        """Transform new texts."""
        all_features = [self._extract_single(t) for t in texts]
        df = pd.DataFrame(all_features, columns=self.feature_names_)
        return csr_matrix(df.values)


class Word2VecExtractor:
    """Extract Word2Vec-based features from text."""

    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.model = None
        self.scaler = StandardScaler()

    def fit_transform(self, texts):
        """Fit Word2Vec and extract features."""
        from gensim.models import Word2Vec

        tokenized = [t.split() for t in texts]
        self.model = Word2Vec(
            tokenized, vector_size=self.vector_size, window=5,
            min_count=1, workers=1, epochs=50, seed=42
        )

        features = self._get_doc_vectors(tokenized)
        self.scaler.fit(features)
        return csr_matrix(self.scaler.transform(features))

    def transform(self, texts):
        """Transform new texts using fitted model."""
        tokenized = [t.split() for t in texts]
        features = self._get_doc_vectors(tokenized)
        return csr_matrix(self.scaler.transform(features))

    def _get_doc_vectors(self, tokenized_texts):
        """Average word vectors for each document."""
        result = np.zeros((len(tokenized_texts), self.vector_size))
        for i, tokens in enumerate(tokenized_texts):
            vectors = []
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
            if vectors:
                result[i] = np.mean(vectors, axis=0)
        return result


class CombinedFeatureExtractor:
    """Combines TF-IDF + metadata + sentiment features."""

    def __init__(self, max_features=5000, ngram_range=(1, 2),
                 use_metadata=True, use_sentiment=True, use_word2vec=False):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_metadata = use_metadata
        self.use_sentiment = use_sentiment
        self.use_word2vec = use_word2vec

        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            sublinear_tf=True, min_df=2, max_df=0.95
        )
        self.metadata = MetadataExtractor() if use_metadata else None
        self.sentiment = SentimentExtractor() if use_sentiment else None
        self.word2vec = Word2VecExtractor() if use_word2vec else None

    def fit_transform(self, texts, raw_texts=None):
        """Fit and transform all feature types."""
        tfidf_features = self.tfidf.fit_transform(texts)
        features = [tfidf_features]

        if self.metadata and raw_texts is not None:
            meta_features = self.metadata.fit_transform(texts, raw_texts)
            features.append(meta_features)

        if self.sentiment:
            sent_features = self.sentiment.fit_transform(texts)
            features.append(sent_features)

        if self.word2vec and raw_texts is not None:
            w2v_features = self.word2vec.fit_transform(texts)
            features.append(w2v_features)

        return hstack(features)

    def transform(self, texts, raw_texts=None):
        """Transform using fitted extractors."""
        tfidf_features = self.tfidf.transform(texts)
        features = [tfidf_features]

        if self.metadata and raw_texts is not None:
            meta_features = self.metadata.transform(texts, raw_texts)
            features.append(meta_features)

        if self.sentiment:
            sent_features = self.sentiment.transform(texts)
            features.append(sent_features)

        if self.word2vec and raw_texts is not None:
            w2v_features = self.word2vec.transform(texts)
            features.append(w2v_features)

        return hstack(features)

    def get_feature_names(self):
        """Return all feature names."""
        names = list(self.tfidf.get_feature_names_out())
        if self.metadata:
            names.extend(self.metadata.feature_names_)
        if self.sentiment:
            names.extend(self.sentiment.feature_names_)
        return names


# Keep original FeatureExtractor for backward compatibility
class FeatureExtractor:
    """Extracts numerical features from preprocessed text (original API)."""

    def __init__(self, method='tfidf', max_features=10000, ngram_range=(1, 2)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None

    def fit_transform(self, texts):
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, ngram_range=self.ngram_range,
                sublinear_tf=True, min_df=2, max_df=0.95
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features, ngram_range=self.ngram_range,
                min_df=2, max_df=0.95
            )
        else:
            raise ValueError(f"Unknown method: {self.method}.")
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer not fitted.")
        return self.vectorizer.get_feature_names_out()

    def get_top_features(self, n=20):
        if self.vectorizer is None or self.method != 'tfidf':
            return []
        feature_names = self.get_feature_names()
        idf_scores = self.vectorizer.idf_
        top_indices = idf_scores.argsort()[:n]
        return [(feature_names[i], idf_scores[i]) for i in top_indices]
