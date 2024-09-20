import os
import csv
import sys
from PIL import Image

#from dasr_vae import spectrogram_dir
import numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Certifique-se de importar corretamente
import math
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import datetime
import keras
from keras import layers
from keras.models import load_model
print(tf.__version__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
tf.config.optimizer.set_jit(False)  # Desativa o XLA JIT

# Lista todos os dispositivos físicos do tipo 'GPU'
devices = tf.config.list_physical_devices('GPU')
print(len(devices))  # Se o resultado for maior que 0, uma GPU está sendo usada

# Verifica se o TensorFlow foi construído com suporte a CUDA
print(tf.test.is_built_with_cuda())
'''
# Função para converter string para booleano
def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("A string deve ser 'True' ou 'False'")

# Verifica se os argumentos foram fornecidos
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python asr_trafo.py <diretório_da_base_de_dados> <tipo_da_base: original, aug, aug_prop, aug_plus_prop><ffn do ultimo decoder treinável> <ultimo decoder treinavel> <epocas> <arquivo pre treino>")
        sys.exit(1)

    base = sys.argv[1]
    tipo_base = sys.argv[2]
    fz_last_dec_ffn = str_to_bool(sys.argv[3])
    fz_last_dec = str_to_bool(sys.argv[4])
    epocas = int(sys.argv[5])

    if len(sys.argv) > 6:
        pre_train = sys.argv[6]
    else:
        pre_train = ""
'''

fz_last_dec_ffn = True
fz_last_dec = True
epocas = 200
pre_train = ""
batch_size = 64
bs = batch_size

# Função para ler as transcrições
def read_transcript(file_path):
    id_to_text = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                file_id, transcript = parts
                id_to_text[file_id] = transcript
    return id_to_text

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
'''
# Função para carregar um espectrograma a partir de uma imagem
def load_spectrogram(file, target_size=(2754, 256)):
    # Carregar a imagem completa
    img = Image.open(file)
    
    # Define as coordenadas da parte direita (reconstruído)
    width, height = img.size
    left = width // 2  # Metade da largura para pegar o gráfico da direita
    upper = 0
    right = width
    lower = height
    
    # Corta a parte reconstruída (lado direito)
    img_reconstructed = img.crop((left, upper, right, lower))

    # Converte para escala de cinza
    img_reconstructed_gray = img_reconstructed.convert('L')

    # Converte para array e normaliza
    img_array = img_to_array(img_reconstructed_gray)
    img_array /= 255.0

    current_shape = img_array.shape

    # Calcula padding para o target_size
    pad_height = max(target_size[0] - current_shape[0], 0)
    pad_width = max(target_size[1] - current_shape[1], 0)

    # Aplica padding se necessário
    img_array = np.pad(img_array, 
                       pad_width=((0, pad_height), (0, pad_width), (0, 0)), 
                       mode='constant', constant_values=0)

    # Garante que o array tenha o tamanho desejado
    #img_array = img_array[:target_size[0], :target_size[1]]

    # Achatar a dimensão 2D para que seja compatível com a convolução 1D
    img_array = np.reshape(img_array, (target_size[0], -1)) 
    # Retorna o array em uma forma pronta para convoluções 1D
    return img_array

# Função para carregar os espectrogramas em lotes
def get_reconstructed_spectrograms(directory, target_size=(2754, 256), batch_size=32):
    # Verifica se target_size é uma tupla
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # Lista arquivos
    reconstructed_files = sorted(glob(os.path.join(directory, '*.png')))

    # Função geradora para o dataset
    def data_generator(files, batch_size):
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_spectrograms = [load_spectrogram(file, target_size) for file in batch_files]
            yield np.array(batch_spectrograms)

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(reconstructed_files, batch_size),
        output_signature=tf.TensorSpec(shape=(None, target_size[0], target_size[1] * 1), dtype=tf.float32)  # Combina altura e largura
    )
  
    file_ids = [os.path.basename(file) for file in reconstructed_files]
    
    steps_per_epoch = len(reconstructed_files) // batch_size
    
    return dataset, file_ids, steps_per_epoch
'''

def load_spectrogram(file, target_shape=(2754, 129)):
    # Carregar a imagem completa
    img = Image.open(file)
    
    # Converte para escala de cinza
    img_reconstructed_gray = img.convert('L')

    # Converte para array e normaliza
    img_array = img_to_array(img_reconstructed_gray)
    img_array /= 255.0

    # Redimensiona para o target_shape
    img_array = np.resize(img_array, target_shape)

    # Adiciona a dimensão de canais (1 canal para grayscale)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def pad_batch_to_size(batch, batch_size):
    # Verificar se o tamanho do batch é menor que o batch_size
    current_size = batch.shape[0]
    
    if current_size < batch_size:
        # Número de duplicações necessárias
        num_duplicates = batch_size - current_size
        
        # Escolher aleatoriamente amostras para duplicação
        indices_to_duplicate = np.random.choice(current_size, num_duplicates, replace=True)
        
        # Adicionar as amostras duplicadas ao batch
        duplicates = batch[indices_to_duplicate]
        batch = np.concatenate([batch, duplicates], axis=0)
    
    return batch

# Modificação no generator
def data_generator(files, batch_size, target_shape=(2754, 129)):
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_spectrograms = []
        for file in batch_files:
            spectrogram = load_spectrogram(file, target_shape)
            batch_spectrograms.append(spectrogram)
        
        yield np.array(batch_spectrograms)  # Saída será (batch_size, height, width, 1)

# Modificação no dataset
def get_reconstructed_spectrograms(directory, target_size=(2754, 129), batch_size=64):
    # Verifica se target_size é uma tupla
    if isinstance(target_size, int):
        target_size = (target_size, target_size)  # Converte para tupla (altura, largura)

    # Lista arquivos
    reconstructed_files = sorted(glob(os.path.join(directory, '*.png')))

    # Função geradora para o dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(reconstructed_files, batch_size, target_size),
        output_signature=tf.TensorSpec(shape=(batch_size, target_size[0], target_size[1], 1), dtype=tf.float32)
    )

    steps_per_epoch = len(reconstructed_files) // batch_size
    file_ids = [os.path.basename(file) for file in reconstructed_files]

    return dataset, file_ids, steps_per_epoch


reconstruct_base_dir = 'D:/POS/auto_dasr/dataset/reconstruct'
spectrogram_dir = os.path.join(reconstruct_base_dir, 'spectrograms')
# Carrega as transcrições
transcript_file = os.path.join(spectrogram_dir, "ua_transcript.txt")
id_to_text = read_transcript(transcript_file)

# Carrega os espectrogramas reconstruídos
reconstructed_ds, file_ids, steps_per_epoch = get_reconstructed_spectrograms(directory=spectrogram_dir, target_size=200, batch_size=bs)
#steps_per_epoch = math.ceil(len(file_ids) // bs)

# Define o embedding para tokens
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

# Define o embedding para features de áudio (neste caso, espectrogramas)
class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        # Remover a dimensão do canal
        x = tf.squeeze(x, axis=-1)  # Agora x terá o shape (batch_size, 2754, 129)
        
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)



# Define o Transformer Encoder
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                #layers.Dense(feed_forward_dim, activation="relu"),
                #layers.Dense(embed_dim),
                layers.SeparableConv1D(feed_forward_dim, 3, activation='relu', padding='same'),
                layers.SeparableConv1D(embed_dim, 3, padding='same'),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=True):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ## Transformer Decoder Layer

# In[90]:


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu", trainable=True),
                layers.Dense(embed_dim, trainable=True),
            ]
        )
        self.ffn.trainable = True

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


# ## Modelo Transformer completo

class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=5,
        num_layers_dec=3,
        num_classes=34,
    ):
        super().__init__()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = tf.keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )
        # Congelando as Ãºltimas camadas do codificador
        #for layer in self.encoder.layers[-2:]:
        #    layer.trainable = False
        #    if layer.trainable:
        #        print(f"Camada {layer} do Encoder: Descongelada")
        #    else:
        #        print(f"Camada {layer} do Encoder: Congelada")


          

        for i in range(num_layers_dec):
            layer_dec = TransformerDecoder(num_hid, num_head, num_feed_forward)
            layer_dec.trainable = True
            if i == num_layers_dec - 1:  # Se for a última camada
                layer_dec.ffn.trainable = fz_last_dec_ffn  # Congela/desconhgela o componente feed forward
                print(f"Componente FFN da camada decodificadora {i} congelado: {not layer_dec.ffn.trainable}")   
            setattr(
                self,
                f"dec_layer_{i}",
                layer_dec,
            )
            if i == num_layers_dec - 1:  # Se for a última camada
                layer_dec.trainable = fz_last_dec  # Congela/descongela a ultima camada decodificadora
            print(f"Camada decodificadora {i} congelada: {not layer_dec.trainable}")   
            
        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        
        input_shape = tf.shape(source)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]

        # Achatar as dimensões de height e width para combinar em uma única dimensão de features
        source = tf.reshape(source, (batch_size, seq_len, height * width))

        print(f"Forma da entrada no encoder: {source.shape}")
        x = self.encoder(source)
        print(f"Forma da saída do encoder: {x.shape}")
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            loss = loss_object(one_hot, preds, sample_weight=mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        loss = loss_object(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

# Funções auxiliares
class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

def create_text_ds(texts, vectorizer):
    text_ds = [vectorizer(text) for text in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds

def create_audio_ds(reconstructed_ds):
    #audio_ds = tf.data.Dataset.from_tensor_slices(reconstructed_ds)
    return reconstructed_ds

def create_tf_dataset(reconstructed_ds, texts, vectorizer, bs=bs):
    # Garantir que reconstructed_ds tenha a forma (batch_size, 256, 256)
    audio_ds = reconstructed_ds  # O reconstructed_ds já deve estar no formato correto
    
    # Processar o texto
    text_ds = create_text_ds(texts, vectorizer)
    
    # Juntar os dois datasets (áudio e transcrições)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    
    # Mapear para o formato adequado
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    
    # Aplicar batch e prefetch para eficiência
    ds = ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    
    return ds


# Associa os espectrogramas reconstruídos com as transcrições
def match_spectrograms_with_transcripts(file_ids, id_to_text):
    texts = [id_to_text.get(file_id, "") for file_id in file_ids]
    return texts

# Criação do Dataset
vectorizer = VectorizeChar(max_len=200)
texts = match_spectrograms_with_transcripts(file_ids, id_to_text)
ds = create_tf_dataset(reconstructed_ds, texts, vectorizer, bs=bs)
val_ds = create_tf_dataset(reconstructed_ds, texts, vectorizer, bs=bs)

# Configuração do modelo
model = Transformer(
    num_hid=200,
    num_head=2,
    num_feed_forward=400,
    target_maxlen=200,
    num_layers_enc=5,
    num_layers_dec=3,
    num_classes=len(vectorizer.get_vocabulary()),
)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, lr_after_warmup, final_lr, warmup_epochs, decay_epochs, steps_per_epoch):
        super(CustomSchedule, self).__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = decay_epochs * steps_per_epoch

    def __call__(self, step):
        # Fase de warmup
        lr = tf.cond(
            tf.cast(step, tf.float32) < tf.cast(self.warmup_steps, tf.float32),
            lambda: tf.cast(self.init_lr * tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32), tf.float32),
            lambda: tf.cast(self.final_lr + (self.lr_after_warmup - self.final_lr) * tf.math.exp(-0.001 * tf.cast(step - self.warmup_steps, tf.float32)), tf.float32)
)
        return lr

    def get_config(self):
        return {
            "init_lr": self.init_lr,
            "lr_after_warmup": self.lr_after_warmup,
            "final_lr": self.final_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
learning_rate = CustomSchedule(
    init_lr=0.00001,
    lr_after_warmup=0.001,
    final_lr=0.00001,
    warmup_epochs=15,
    decay_epochs=85,
    #steps_per_epoch=len(ds),
    steps_per_epoch = steps_per_epoch,
)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.8, beta_2=0.9)

if pre_train:
    model.load_weights(pre_train)

model.compile(optimizer=optimizer, loss=loss_fn)

# Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=5, write_graph=True, write_images=True)

checkpoint = ModelCheckpoint('melhor_modelo.ckpt',
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)

# Treinamento do modelo
history = model.fit(ds, validation_data=val_ds, epochs=epocas, steps_per_epoch=steps_per_epoch,  callbacks=[tensorboard_callback, checkpoint])

model.summary()

# Cálculo da acurácia por palavra
def compute_word_accuracy(model, val_ds, idx_to_char):
    total_words = 0
    correct_words = 0
    
    for batch in val_ds:
        source = batch["source"]
        target = batch["target"]
        preds = model.generate(source, target_start_token_idx=2)
        preds = preds.numpy()
        for i in range(len(source)):
            target_text = "".join([idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += idx_to_char[idx]
                if idx == 3:
                    break
            target_words = target_text.replace('-', '').split()
            prediction_words = prediction.split()
            total_words += len(target_words)
            correct_words += sum(1 for tw, pw in zip(target_words, prediction_words) if tw == pw)

    accuracy = correct_words / total_words if total_words > 0 else 0
    return accuracy

model.load_weights('pre_ua/melhor_modelo.ckpt')
idx_to_char = vectorizer.get_vocabulary()
word_accuracy = compute_word_accuracy(model, val_ds, idx_to_char)
print("Word Accuracy:", word_accuracy)

with open("resultados.txt", "a") as file:
    file.write(f"{base}_{tipo_base}_| Acurácia: {word_accuracy}\n")
