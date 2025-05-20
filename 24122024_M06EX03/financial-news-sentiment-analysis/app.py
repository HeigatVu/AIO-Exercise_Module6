import streamlit as st
import torch
import torch.nn as nn
# from torchtext.data.utils import get_tokenizer # Not needed if using .split() as per your vocab generation
import pickle
import nltk
import unidecode
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Configuration ---
VOCAB_PATH = 'vocab.pkl'
MODEL_PATH = './model/rnn_sentimentclassifier.pth' # Ensure this path is correct
MAX_SEQ_LEN = 32 # Must match training

# --- Efficient Resource Loading & Initialization ---

# Download nltk data only if necessary
@st.cache_resource
def download_nltk_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    return True

download_nltk_stopwords()

@st.cache_resource
def get_preprocessing_tools():
    stop_words = set(stopwords.words('english')) # Use set for faster lookups
    stemmer_instance = PorterStemmer()
    return stop_words, stemmer_instance

english_stop_words, stemmer = get_preprocessing_tools()

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, n_classes, dropout_prob):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x) # Use _ for hn as it's not used
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@st.cache_resource # Cache the model and vocab loading
def load_model_and_vocab():
    with open(VOCAB_PATH, 'rb') as f:
        vocab_list = pickle.load(f) # vocab.pkl stores a list of words
    # Create word_to_idx from the loaded list
    word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}

    # Ensure 'UNK' and 'PAD' are in the vocabulary (critical)
    if 'UNK' not in word_to_idx:
        st.error(f"Critical Error: 'UNK' token not found in loaded vocabulary from {VOCAB_PATH}.")
        # Potentially add it if you know its intended index or stop execution
    if 'PAD' not in word_to_idx:
        st.error(f"Critical Error: 'PAD' token not found in loaded vocabulary from {VOCAB_PATH}.")
        # Potentially add it or stop execution

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters (must match training)
    vocab_size = len(word_to_idx) # Size of the loaded vocabulary
    embedding_dim = 64
    hidden_size = 64
    n_layers = 2
    n_classes = 3 # IMPORTANT: Verify this matches your training data's unique sentiment classes

    model_instance = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_classes=n_classes,
        dropout_prob=0, # Set to 0 for evaluation
    ).to(device)

    model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_instance.eval()
    return model_instance, word_to_idx, device

model, vocab, device = load_model_and_vocab()

# --- Text Preprocessing Function (Consistent with Training) ---
def text_normalize_and_tokenize(text_input):
    # 1. Normalize
    text = text_input.lower()
    text = unidecode.unidecode(text)
    text = text.strip()
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation

    # 2. Tokenize using .split() (CRITICAL: This must match vocab creation)
    raw_tokens = text.split(' ')

    # 3. Remove stopwords and stem
    processed_tokens = [
        stemmer.stem(word) for word in raw_tokens if word and word not in english_stop_words
    ]
    return processed_tokens

# --- Streamlit App UI ---
st.title('Financial News Sentiment Analysis')
st.markdown("## Input News")
text_input = st.text_area("Enter your financial news text here:", height=200)

if st.button('Analyze Sentiment'):
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    elif 'UNK' not in vocab or 'PAD' not in vocab:
        st.error("Vocabulary not loaded correctly. 'UNK' or 'PAD' tokens missing. Cannot proceed.")
    else:
        # 1. Preprocess text: Normalize, tokenize (using .split()), stem, remove stopwords
        tokens = text_normalize_and_tokenize(text_input)

        # 2. Convert tokens to indices
        unk_idx = vocab['UNK']
        pad_idx = vocab['PAD']
        indices = [vocab.get(token, unk_idx) for token in tokens]

        # 3. Pad or truncate sequence
        if len(indices) < MAX_SEQ_LEN:
            indices.extend([pad_idx] * (MAX_SEQ_LEN - len(indices)))
        else:
            indices = indices[:MAX_SEQ_LEN]

        # 4. Convert to tensor
        # Use torch.tensor with dtype=torch.long for modern PyTorch
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

        # 5. Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities_tensor = torch.nn.functional.softmax(output, dim=1)[0]
            prediction_idx = torch.argmax(probabilities_tensor).item() # Argmax on probabilities or logits is same

        # 6. Map prediction to sentiment
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'} # Verify this mapping!
        sentiment = sentiment_map.get(prediction_idx, "Unknown")
        predicted_probability = probabilities_tensor[prediction_idx].item()

        st.markdown(f"## Sentiment: **{sentiment}** with **{predicted_probability:.2%}** probability")

        # Optional: Display token and index information for debugging
        with st.expander("See tokenization details"):
            st.write(f"Processed Tokens: {tokens}")
            st.write(f"Token Indices: {indices}")
            st.write(f"Input Tensor: {input_tensor}")