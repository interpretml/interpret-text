import pandas as pd
from typing import Any, Iterable

from interpret_text.common.base_explainer import BaseTextPreprocessor
from transformers import BertTokenizer


class BertPreprocessor(BaseTextPreprocessor):
    
    MAX_TOKEN_LENGTH = 50  # change to param?
    TOKEN_PAD_TO_MAX = True 
    BUILD_WORD_DICT = True  # Default = False?
    
    def build_vocab(self, text: str) -> Any:
        words_to_indexes = {}
        counts = {}
        tokenizer = self.get_tokenizer()
        
        for special_token in tokenizer.all_special_tokens:
            words_to_indexes[special_token] = tokenizer.convert_tokens_to_ids(special_token)
            counts[special_token] = 1000
        print(words_to_indexes)

        for sentence in text:
            tokens = self.generate_tokens(sentence)
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            for (token, token_id) in zip(tokens, ids):
                if token not in words_to_indexes:
                    words_to_indexes[token] = token_id
                    counts[token] = 1
                else:
                    counts[token] += 1

        indexes_to_words = {v: k for k, v in words_to_indexes.items()}

        max_num = max(words_to_indexes.values())

        for i in range(max_num):
            if i not in indexes_to_words:
                token = tokenizer.convert_ids_to_tokens(i)
                indexes_to_words[i] = token
                words_to_indexes[token] = i
                counts[token] = 0

        print('max num:', max_num)
        print("vocab size:", len(words_to_indexes))

        self.word_vocab = words_to_indexes
        self.reverse_word_vocab = indexes_to_words
        self.counts = counts

    def preprocess(self, data: Iterable[Any]) -> Iterable[Any]:
        input_ids = []
        attention_mask = []
        counts = []
        
        tokenizer = self.get_tokenizer()
        for text in data:
            d = tokenizer.encode_plus(
                text,
                max_length=self.MAX_TOKEN_LENGTH,
                pad_to_max_length=self.TOKEN_PAD_TO_MAX,
            )
            input_ids.append(d["input_ids"])
            attention_mask.append(d["attention_mask"])
            counts.append(sum(d["attention_mask"]))

        tokens = pd.DataFrame(
            {"tokens": input_ids, "mask": attention_mask, "counts": counts}
        )

        return tokens

    def decode_single(self, id_list) -> Iterable[Any]:
        tokenizer = self.get_tokenizer()
        return tokenizer.convert_ids_to_tokens(id_list)

    def generate_tokens(self, sentence: str) -> Iterable[Any]:
        return self.get_tokenizer().tokenize(sentence)

    def get_tokenizer(self) -> BertTokenizer:
        return BertTokenizer.from_pretrained("bert-base-uncased")
