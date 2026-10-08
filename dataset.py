import os
import torch
import urllib.request

SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"

def download_shakespeare():
    if os.path.exists(DATA_PATH):
        print("Data file exists, skipping")
        return

    os.makedirs(name="data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)


# Tokenizer işlemini "hello" -> [7,2,11,11,9] şeklinde her harfe bir sayı gelecek şekilde kendimiz yazacağız.

class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(list(set(text)))

        self.vocab_size = len(self.characters)

        self.char_to_id = {}

        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index

        self.id_to_char = {}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char

        print(f"Vocab Size: {self.vocab_size}")
        print(f"Characters: {self.characters}")

    def encode(self, text: str) -> list:
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

    def decode(self, ids: list) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        characters = []
        for id in ids:
            characters.append(self.id_to_char[id])
        return ''.join(characters)

def load_data(train_split: float = 0.9):
    download_shakespeare()
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = CharacterTokenizer(text)

    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    split_index = int(len(data) * train_split)
    train_data = data[:split_index]
    test_data = data[split_index:]
    print(f"Train data size : {len(train_data)}")
    print(f"Test data size : {len(test_data)}")

    return train_data, test_data, tokenizer

def get_batch(data: torch.tensor, block_size: int, batch_size: int):
    # "To be or not to be"
    # "to b" -> x
    # "to be" -> y
    # x = ["T", "o", " ", "b", "e"]
    # y = ["o", " ", "b", "e", " "]

    # x = ["o", " ", "b", "e", " "]
    # y = [" ", "b", "e", " ", "o"]

    # block_size -> sequence length
    # batch_size -> # of sequence

    max_start = len(data) - block_size - 1   # ilerlerken en fazla nereye kadar gidebileceği (-1 y'den dolayı oluştu)
    positions = torch.randint(max_start, (batch_size, ))

    x_list = []
    y_list = []

    for pos in positions:
        x_list.append(data[pos : pos + block_size])
        y_list.append(data[pos + 1 : pos + block_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    return x, y


if __name__ == "__main__":

    train_data, test_data, tokenizer = load_data()
    x, y = get_batch(train_data, block_size=256, batch_size=5)
    print(x.shape)
    print(y.shape)
