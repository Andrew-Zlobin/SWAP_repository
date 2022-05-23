import jsonlines

def get_dataset(path='RUSSE\\train.jsonl', size=10):
    text = ""
    counter = 0
    with jsonlines.open(path) as reader:
        for obj in reader:
            counter += 1
            print(counter)
            if counter >= size:
                break

            text = text + obj["sentence1"] + " " + obj["sentence2"] + " "
    print(text)
    return text


if __name__ == "__main__":
    get_dataset()