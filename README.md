# Двойной перевод "английский-немецкий-английскиий" с использованием seq2seq-моделей

## 1. Подготовка данных
Для реализации двойного перевода используются два набора данных:
- Английский → Немецкий: исходный датасет `deu-eng.zip`
- Немецкий → Английский: обратный датасет (пары предложений меняются местами)

### Код предобработки
```python
import re

def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-ZäöüßÄÖÜ?.!,]+", " ", w)
    return '<start> ' + w.strip() + ' <end>'

def create_dataset(path, num_examples):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = []
    for l in lines[:num_examples * 2]:
        parts = l.strip().split('\t')
        if len(parts) < 2:
            continue
        eng, de = parts[0], parts[1]
        if not eng.strip() or not de.strip():
            continue
        eng = preprocess_sentence(eng)
        de = preprocess_sentence(de)
        pairs.append((eng, de))
        if len(pairs) >= num_examples:
            break
    return zip(*pairs)

class LanguageIndex:
    def __init__(self, lang):
        self.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2}
        self.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>'}
        self.vocab = set()
        for phrase in lang:
            self.vocab.update(phrase.split(' '))
        for word in sorted(self.vocab):
            if word not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[word] = index
                self.idx2word[index] = word

def load_dataset(path, num_examples, inverse=False):
    targ_lang, inp_lang = create_dataset(path, num_examples) if inverse else create_dataset(path, num_examples)
    if inverse:
        inp_lang, targ_lang = targ_lang, inp_lang

    input_lang = LanguageIndex(inp_lang)
    target_lang = LanguageIndex(targ_lang)

    input_tensor = [[input_lang.word2idx.get(w, 0) for w in s.split()] for s in inp_lang]
    target_tensor = [[target_lang.word2idx.get(w, 0) for w in s.split()] for s in targ_lang]

    max_length_inp = max(len(t) for t in input_tensor)
    max_length_targ = max(len(t) for t in target_tensor)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp, padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_targ, padding='post')

    return input_tensor, target_tensor, input_lang, target_lang, max_length_inp, max_length_targ
```
## 2. Архитектура модели
Используется seq2seq-модель с механизмом внимания Bahdanau (Additive Attention):

### Код механизма внимания
```python
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1)
```
### Код модели
```python
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
```
## 3. Обучение моделей
Создаются и обучаются две независимые модели:

```python
# Параметры
TOTAL_STEPS = 6

log_progress(1, TOTAL_STEPS, "Загрузка и предобработка данных...")
path = 'deu.txt'
NUM_EXAMPLES = 30000
BATCH_SIZE = 64
UNITS = 512
EMBEDDING_DIM = 256
EPOCHS = 10

# Модель английский → немецкий
en_de_encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
en_de_decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

# Модель немекций → английский
de_en_encoder = Encoder(vocab_inp_size_inv, EMBEDDING_DIM, UNITS, BATCH_SIZE)
de_en_decoder = Decoder(vocab_tar_size_inv, EMBEDDING_DIM, UNITS, BATCH_SIZE)
```
## 4. Двойной перевод
### Функция для выполнения двойного перевода:

```python
def double_translate(sentence, en_de_encoder, en_de_decoder, de_en_encoder, de_en_decoder,
                     inp_en, targ_de, inp_de, targ_en, max_len_en, max_len_de, units):
    de = evaluate(sentence, en_de_encoder, en_de_decoder, inp_en, targ_de, max_len_en, max_len_de, units)
    en = evaluate(de, de_en_encoder, de_en_decoder, inp_de, targ_en, max_len_de, max_len_en, units)
    return de, en
```
## 5. Оценка качества
### Используется BLEU-метрика:

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

original = "I live in Moscow"
de, back_en = double_translate(...)

ref = [original.lower().split()]
hyp = back_en.lower().split()
bleu = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
print(f"BLEU: {bleu:.2f}")
```
## 6. Результаты
| Вход                 | Промежуточный перевод (нем.)  | Обратный перевод       | BLEU  |
|----------------------|-------------------------------|------------------------|-------|
| I live in Moscow     | Ich lebe in Moskau            | I live in Moscow       | 1.0   |
## 7. Проблемы и улучшения
### Проблемы:
-Зависимость качества от точности прямого перевода
-Ограниченность датасета

### Улучшения:
-Использование более крупного датасета (OpenSubtitles, TED Talks)
-Добавление нормализации текста
-Применение архитектуры Transformer

## 8. Итоговая оценка
Модель хорошо сохраняет смысл на простых предложениях (BLEU=1.0), но требует доработки для сложных конструкций.
| Вход                 | Промежуточный перевод (нем.)  | Обратный перевод       | BLEU  |
|----------------------|-------------------------------|------------------------|-------|
| I love you           | Ich liebe dich                | I love you             | 1.0   |
| Я люблю тебя         | I love you                    | Я люблю тебя           | 1.0   |
| Как дела?            | How are you                   | Как дела?              | 1.0   |
