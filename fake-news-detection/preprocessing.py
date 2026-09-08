"""
Text Preprocessing Module
Handles cleaning and normalization of text data for fake news detection.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class TextPreprocessor:
    """Handles all text preprocessing operations."""

    def __init__(self, use_stemming=True, use_lemmatization=False,
                 remove_stopwords=True, lowercase=True):
        """
        Initialize the preprocessor.

        Args:
            use_stemming: Apply Porter Stemmer to reduce words to root form.
            use_lemmatization: Apply WordNet Lemmatizer (slower but more accurate).
            remove_stopwords: Remove common English stop words.
            lowercase: Convert all text to lowercase.
        """
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase

        if use_stemming:
            self.stemmer = PorterStemmer()
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Apply all preprocessing steps to a text string.

        Args:
            text: Raw text string to preprocess.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        words = text.split()

        # Remove stop words
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words and len(w) > 2]

        # Stemming
        if self.use_stemming:
            words = [self.stemmer.stem(w) for w in words]

        # Lemmatization
        if self.use_lemmatization:
            words = [self.lemmatizer.lemmatize(w) for w in words]

        return ' '.join(words)

    def preprocess_dataframe(self, df, text_column='text', title_column=None):
        """
        Preprocess text columns in a pandas DataFrame.

        Args:
            df: Input DataFrame.
            text_column: Name of the main text column.
            title_column: Optional name of a title column to combine with text.

        Returns:
            DataFrame with added 'processed_text' column.
        """
        df = df.copy()

        # Combine title and text if title column exists
        if title_column and title_column in df.columns:
            df['combined_text'] = df[title_column].fillna('') + ' ' + df[text_column].fillna('')
        else:
            df['combined_text'] = df[text_column].fillna('')

        # Apply preprocessing
        df['processed_text'] = df['combined_text'].apply(self.clean_text)

        # Remove empty rows
        df = df[df['processed_text'].str.strip().astype(bool)].reset_index(drop=True)

        return df
