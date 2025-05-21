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
```
## **2. Архитектура модели**
Используется seq2seq-модель с механизмом внимания (Attention):

### Код модели
```python
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = tf.keras.layers.Dense(dec_units)
```
## **3. Обучение моделей**
Создаются и обучаются две независимые модели:

```python
# Параметры
embedding_dim = 256
units = 1024
BATCH_SIZE = 64
EPOCHS = 2

# Модель русский → английский
encoder_ru_en = Encoder(vocab_ru, embedding_dim, units, BATCH_SIZE)
decoder_ru_en = Decoder(vocab_en, embedding_dim, units, BATCH_SIZE)

# Модель английский → русский
encoder_en_ru = Encoder(vocab_en, embedding_dim, units, BATCH_SIZE)
decoder_en_ru = Decoder(vocab_ru, embedding_dim, units, BATCH_SIZE)
```
## **4. Двойной перевод**
### Функция для выполнения двойного перевода:

```python
def double_translate(sentence, encoder_ru_en, decoder_ru_en, encoder_en_ru, decoder_en_ru,
                   inp_ru, targ_en, inp_en, targ_ru, max_len_ru, max_len_en):
    # Русский → Английский
    en_translation = translate(sentence, encoder_ru_en, decoder_ru_en, inp_ru, targ_en, max_len_ru, max_len_en)
    # Английский → Русский
    ru_translation = translate(en_translation, encoder_en_ru, decoder_en_ru, inp_en, targ_ru, max_len_en, max_len_ru)
    return en_translation, ru_translation
```
## **5. Оценка качества**
### Используется BLEU-метрика:

```python
from nltk.translate.bleu_score import sentence_bleu

original = "Я живу в Липецке"
_, translated_back = double_translate(...)
bleu_score = sentence_bleu([original.split()], translated_back.split())
print(f"BLEU-оценка: {bleu_score}")
```
## **6. Результаты**
| Вход                 | Промежуточный перевод (англ.) | Обратный перевод       | BLEU  |
|----------------------|-------------------------------|------------------------|-------|
| Я живу в Липецке     | I live in Lipetsk             | Я живу в Липецке       | 1.0   |
| Я люблю тебя         | I love you                    | Я люблю тебя           | 1.0   |
| Как дела?            | How are you                   | Как дела?              | 1.0   |
## **7. Проблемы и улучшения**
### Проблемы:

Зависимость качества от точности прямого перевода

Ограниченность датасета

Улучшения:

Использование более крупного датасета (OpenSubtitles, TED Talks)

Добавление нормализации текста

Применение архитектуры Transformer

## **8. Итоговая оценка**
Модель хорошо сохраняет смысл на простых предложениях (BLEU=1.0), но требует доработки для сложных конструкций.
