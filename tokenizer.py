import re

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
  for i, item in enumerate(vocab.items()):
      print(item)
      if i >= 50:
          break