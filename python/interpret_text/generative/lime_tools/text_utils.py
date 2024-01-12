import spacy

nlp = spacy.load("en_core_web_sm")


def extract_non_overlapping_ngrams(text, n):
    # Tokenize the text into words or tokens
    tokens = text.split()
    # Extract non-overlapping n-grams
    non_overlapping_ngrams = [tokens[i:i + n]
                              for i in range(0, len(tokens), n)]
    # Convert each group of words into a tuple (to resemble the output of the
    # nltk.ngrams function)
    non_overlapping_ngrams = [" ".join(group)
                              for group in non_overlapping_ngrams]
    return non_overlapping_ngrams


def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def split_k_sentences(text, k):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    combined_sentences = []
    for i in range(0, len(sentences), k):
        combined_sentences.append(' '.join(sentences[i:i + k]))
    return combined_sentences


def split_dep_parse(text, *args, **kwargs):
    doc = nlp(text)
    chunks = []
    chunk = ""
    for token in doc:
        # append the token to the current chunk
        chunk += token.text_with_ws

        # decide whether to chunk at this token
        if token.pos_ in ['PUNCT', 'CCONJ', 'SCONJ']:
            if token.pos_ == 'PUNCT' and len(chunk.split()) == 1:
                continue
            if chunk.strip():
                chunks.append(chunk.strip())
                chunk = ""
        elif token.dep_ == "ROOT":
            if chunk.strip():
                chunks.append(chunk.strip())
                chunk = ""

    # handle the last chunk
    if chunk.strip():
        chunks.append(chunk.strip())

    return chunks
