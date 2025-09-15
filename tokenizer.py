import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) 

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


def load_data():
  with open("data/verdict.txt", "r", encoding="utf-8") as f:
      raw_text = f.read()
  return raw_text


def print_data_statistics(data):
  print("Total number of characters:", len(data))
  print(data[:99])


def regex_tokenize(text):
  pattern = r'([,.:;?_!"()\']|--|\s)'
  result = re.split(pattern, text)
  tokens = [token for token in result if token.strip()]
  return tokens


def create_vocab(tokens):
  all_words = sorted(set(tokens))
  vocab = {token:integer for integer,token in enumerate(all_words)}
  return vocab


if __name__ == "__main__":
  data = load_data()
  print_data_statistics(data)
  preprocessed = regex_tokenize(data)
  vocab = create_vocab(preprocessed)
  tokenizer = SimpleTokenizerV1(vocab)
  text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
  ids = tokenizer.encode(text)
  print(ids)
  print(tokenizer.decode(ids))