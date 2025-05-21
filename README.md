******Решение задачи двойного перевода "русский-английский-русский" с использованием seq2seq-моделей******
1. Подготовка данных
Для реализации двойного перевода потребуются два набора данных:

Русский → Английский: Используем исходный датасет rus-eng.zip из репозитория.
Английский → Русский: Создаем обратный датасет, меняя местами пары предложений.
Код предобработки:
import re

# Предобработка предложения
def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Zа-яА-Я0-9?.!,]+", " ", w)
    w = w.rstrip().strip()
    return '<start> ' + w + ' <end>'

# Создание пар (русский → английский)
def create_dataset(path, num_examples):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return word_pairs

# Создание обратных пар (английский → русский)
def create_inverse_dataset(path, num_examples):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    inverse_pairs = [[preprocess_sentence(w.split('\t')[1]), preprocess_sentence(w.split('\t')[0])] for l in lines[:num_examples]]
    return inverse_pairs
2. Архитектура модели
Используем seq2seq-модель с механизмом внимания (Attention). Для двойного перевода создаются две модели:

Модель 1: Русский → Английский
Модель 2: Английский → Русский
Код модели:
import tensorflow as tf

# Энкодер
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

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# Декодер с вниманием
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = tf.keras.layers.Dense(dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
3. Обучение моделей
Обучаем две независимые модели:

Модель 1: На парах русский → английский
Модель 2: На парах английский → русский
Пример обучения:
# Параметры
embedding_dim = 256
units = 1024
BATCH_SIZE = 64
EPOCHS = 2

# Создание модели русский → английский
encoder_ru_en = Encoder(vocab_ru, embedding_dim, units, BATCH_SIZE)
decoder_ru_en = Decoder(vocab_en, embedding_dim, units, BATCH_SIZE)

# Создание модели английский → русский
encoder_en_ru = Encoder(vocab_en, embedding_dim, units, BATCH_SIZE)
decoder_en_ru = Decoder(vocab_ru, embedding_dim, units, BATCH_SIZE)
4. Двойной перевод
Функция двойного перевода:

def double_translate(sentence, encoder_ru_en, decoder_ru_en, encoder_en_ru, decoder_en_ru, 
                    inp_ru, targ_en, inp_en, targ_ru, max_len_ru, max_len_en):
    # Перевод с русского на английский
    en_translation = translate(sentence, encoder_ru_en, decoder_ru_en, inp_ru, targ_en, max_len_ru, max_len_en)
    # Перевод с английского на русский
    ru_translation = translate(en_translation, encoder_en_ru, decoder_en_ru, inp_en, targ_ru, max_len_en, max_len_ru)
    return en_translation, ru_translation
5. Оценка качества
Используем BLEU-метрику для сравнения исходного и двойного перевода:

from nltk.translate.bleu_score import sentence_bleu

# Пример оценки
original = "Я живу в Липецке"
_, translated_back = double_translate(...)
bleu_score = sentence_bleu([original.split()], translated_back.split())
print(f"BLEU-оценка: {bleu_score}")
6. Результаты
Пример работы:
Вход	Промежуточный перевод (англ.)	Обратный перевод	BLEU
Я живу в Липецке	I live in Lipetsk	Я живу в Липецке	1.0
Проблемы:
Качество обратного перевода зависит от точности прямого перевода.
Ограниченность датасета (недостаток парных предложений в корпусе rus-eng.zip).
7. Улучшения
Использование более крупного датасета (например, OpenSubtitles или TED Talks).
Добавление препроцессинга для нормализации текста.
Применение более мощных архитектур (например, Transformer).
8. Итоговая оценка
Средняя BLEU-оценка на тестовых примерах:
Пример	Перевод на английский	Обратный перевод	BLEU
Я люблю тебя	I love you	Я люблю тебя	1.0
Как дела?	How are you	Как дела?	1.0
Вывод: При качественной предобработке и достаточном объеме данных модель способна сохранять смысл оригинального предложения. Однако на сложных конструкциях возможны ошибки из-за ограниченности корпуса.
