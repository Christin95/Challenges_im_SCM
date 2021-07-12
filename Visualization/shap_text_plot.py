import re
import os
import tensorflow as tf
import numpy as np
import torch
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizerFast, TFBertModel, BertConfig
from alibi.explainers import IntegratedGradients
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam




def decode_sentence(x, reverse_index):
    """Decodes the tokenized sentences from keras IMDB dataset into plain text.
    """
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, 'UNK') for i in x])

def preprocess_reviews(reviews):
    """Preprocess the text.
    """
    REPLACE_NO_SPACE = re.compile("[.;:,!\'?\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

def process_sentences(sentence,
                      tokenizer,
                      max_len):
    """Tokenize the text sentences.
    """
    z = tokenizer(sentence,
                  add_special_tokens = False,
                  padding = 'max_length',
                  max_length = max_len,
                  truncation = True,
                  return_token_type_ids=True,
                  return_attention_mask = True,
                  return_tensors = 'np')
    return z



from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
auto_model_bert = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class AutoModelWrapper(tf.keras.Model):

    def __init__(self, model_bert, **kwargs):
        super().__init__()
        self.model_bert = model_bert

    def call(self, inputs, attention_mask=None):
        out = self.model_bert(inputs,
                              attention_mask=attention_mask)
        return tf.nn.softmax(out.logits)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



auto_model = AutoModelWrapper(auto_model_bert)


max_features = 10000
max_len = 100

z_test_sample = ['I love you, I like you', 'I love you, I like you, but I also kind of dislike you']
z_test_sample = preprocess_reviews(z_test_sample)
z_test_sample = process_sentences(z_test_sample, tokenizer, max_len)


x_test_sample = z_test_sample['input_ids']
kwargs = {k:v for k,v in z_test_sample.items() if k == 'attention_mask'}


auto_model.layers[0].layers

bl = auto_model.layers[0].layers[0].transformer.layer[1]



n_steps = 5
method = "gausslegendre"
internal_batch_size = 5
ig = IntegratedGradients(auto_model,
                          layer=bl,
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)





predictions = auto_model(x_test_sample, **kwargs).numpy().argmax(axis=1)
explanation = ig.explain(x_test_sample,
                         forward_kwargs=kwargs,
                         baselines=None,
                         target=predictions)



# Get attributions values from the explanation object
attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)


attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)



i = 1
x_i = x_test_sample[i]
attrs_i = attrs[i]
pred = predictions[i]
pred_dict = {1: 'Positive review', 0: 'Negative review'}


from IPython.display import HTML


def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"


def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors



words = tokenizer.decode(x_i).split()
colors = colorize(attrs_i)

print('Predicted label =  {}: {}'.format(pred, pred_dict[pred]))

HTML("".join(list(map(hlstr, words, colors))))




def get_embeddings(X_train, model, batch_size=50):

    args = X_train['input_ids']
    kwargs = {k:v for k, v in  X_train.items() if k != 'input_ids'}

    dataset = tf.data.Dataset.from_tensor_slices((args, kwargs)).batch(batch_size)
    dataset = dataset.as_numpy_iterator()

    embbedings = []
    for X_batch in dataset:
        args_b, kwargs_b = X_batch
        batch_embeddings = model(args_b, **kwargs_b)
        embbedings.append(batch_embeddings.last_hidden_state.numpy())

    return np.concatenate(embbedings, axis=0)




print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
test_labels = y_test.copy()
train_labels = y_train.copy()
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

index = imdb.get_word_index()
reverse_index = {value: key for (key, value) in index.items()}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased")
modelBert = TFBertModel.from_pretrained("bert-base-uncased", config=config)

modelBert.trainable=False


X_train, X_test = [], []
for i in range(len(x_train)):
    tr_sentence = decode_sentence(x_train[i], reverse_index)
    X_train.append(tr_sentence)
    te_sentence = decode_sentence(x_test[i], reverse_index)
    X_test.append(te_sentence)




X_train = preprocess_reviews(X_train)
X_train = process_sentences(X_train, tokenizer, max_len)
X_test = preprocess_reviews(X_test)
X_test = process_sentences(X_test, tokenizer, max_len)



train_embbedings = get_embeddings(X_train, modelBert, batch_size=100)

test_embbedings = get_embeddings(X_test, modelBert, batch_size=100)







dropout = 0.1
hidden_dims = 128


class ModelOut(tf.keras.Model):
    def __init__(self, dropout=0.2, hidden_dims=128):
        super().__init__()

        self.dropout = dropout
        self.hidden_dims = hidden_dims

        self.flat = tf.keras.layers.Flatten()
        self.dense_1 =  tf.keras.layers.Dense(hidden_dims, activation='relu')
        self.dropoutl = tf.keras.layers.Dropout(dropout)
        self.dense_2 = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.flat(inputs)
        x = self.dense_1(x)
        x = self.dropoutl(x)
        x = self.dense_2(x)
        return x

    def get_config(self):
        return {"dropout": self.dropout,
                "hidden_dims": self.hidden_dims}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



model_out = ModelOut(dropout=dropout, hidden_dims=hidden_dims)


batch_size = 128
epochs = 3


model_out.fit(train_embbedings, y_train,
              validation_data=(test_embbedings, y_test),
              epochs=epochs,
              batch_size=batch_size,
              verbose=1)



class TextClassifier(tf.keras.Model):
    def __init__(self, model_bert, model_out):
        super().__init__()
        self.model_bert = model_bert
        self.model_out = model_out

    def call(self, inputs, attention_mask=None):
        out = self.model_bert(inputs, attention_mask=attention_mask)
        out = self.model_out(out.last_hidden_state)
        return out

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


text_classifier = TextClassifier(modelBert, model_out)







z_test_sample = [decode_sentence(x_test[i], reverse_index) for i in range(10)]
z_test_sample = preprocess_reviews(z_test_sample)
z_test_sample = process_sentences(z_test_sample, tokenizer, max_len)

x_test_sample = z_test_sample['input_ids']
kwargs = {k:v for k,v in z_test_sample.items() if k == 'attention_mask'}













bl = text_classifier.layers[0].bert.encoder.layer[0]




n_steps = 5
method = "gausslegendre"
internal_batch_size = 5
ig  = IntegratedGradients(text_classifier,
                          layer=bl,
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)





predictions = text_classifier(x_test_sample, **kwargs).numpy().argmax(axis=1)
explanation = ig.explain(x_test_sample,
                         forward_kwargs=kwargs,
                         baselines=None,
                         target=predictions)


attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)





attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)


i = 1
x_i = x_test_sample[i]
attrs_i = attrs[i]
pred = predictions[i]
pred_dict = {1: 'Positive review', 0: 'Negative review'}



words = tokenizer.decode(x_i).split()
colors = colorize(attrs_i)



HTML("".join(list(map(hlstr, words, colors))))





