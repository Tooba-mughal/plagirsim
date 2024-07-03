import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors

def compute_similarity(vector1, vector2):
    return cosine_similarity(vector1, vector2)

# Example texts
text1 = """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."""
text2 = """Machine intelligence, commonly referred to as AI, is the intelligence exhibited by machines, unlike the natural intelligence exhibited by humans and animals."""

# Preprocess the texts
preprocessed_text1 = preprocess(text1)
preprocessed_text2 = preprocess(text2)

# Vectorize the texts
vectors = vectorize_texts([preprocessed_text1, preprocessed_text2])

# Compute similarity
similarity = compute_similarity(vectors[0], vectors[1])[0][0]
print(f"Similarity: {similarity * 100:.2f}%")
