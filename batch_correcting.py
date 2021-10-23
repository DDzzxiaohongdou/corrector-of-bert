import requests
import time
from tqdm import tqdm

URL = 'http://127.0.0.1:5000/test?data='

with open('corpus.txt', 'r', encoding='utf-8') as f:
    list1 = [line.strip() for line in f.readlines()]

list1 = list1[:2000]

print(len(list1))

start = time.time()
for sent in tqdm(list1):
    response = requests.get(URL + sent)
end = time.time()
print('总耗时：', end-start)