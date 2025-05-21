# Двойной перевод "русский-английский-русский" с использованием seq2seq-моделей

## 1. Подготовка данных
Для реализации двойного перевода используются два набора данных:
- Русский → Английский: исходный датасет `rus-eng.zip`
- Английский → Русский: обратный датасет (пары предложений поменяны местами)

### Код предобработки
```python
import re

def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Zа-яА-Я0-9?.!,]+", " ", w)
    w = w.rstrip().strip()
    return '<start> ' + w + ' <end>'

def create_dataset(path, num_examples):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return word_pairs

def create_inverse_dataset(path, num_examples):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    inverse_pairs = [[preprocess_sentence(w.split('\t')[1]), preprocess_sentence(w.split('\t')[0])] 
                   for l in lines[:num_examples]]
    return inverse_pairs
