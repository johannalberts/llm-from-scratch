import re


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int            #1
                        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)    #2
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
  all_words = sorted(list(set(tokens)))
  all_words.extend(["<|endoftext|>", "<|unk|>"])
  vocab = {token:integer for integer,token in enumerate(all_words)}
  return vocab


if __name__ == "__main__":
  data = load_data()
  preprocessed = regex_tokenize(data)
  vocab = create_vocab(preprocessed)
  tokenizer = SimpleTokenizerV2(vocab)
  ids = tokenizer.encode(data)
  print(ids)
  print(tokenizer.decode(ids))