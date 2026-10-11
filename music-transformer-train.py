# %%
# Colab only (not Python outside Colab):
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# IPython magic: cd /content/drive/'My Drive'/MusicTransformer

# %%
# IPython magic: ls

# %%
# Shell escape: !pip install -r requirements.txt

# %%
from model import MusicTransformer
from custom.layers import *
from custom import callback
from tensorflow.python import keras
import params as par
from tensorflow.python import enable_eager_execution
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
from scripts.utils import write_csv
import timeit
tf.executing_eagerly()

# %%
l_r = 0.0024 #@param {type:"slider", min:0, max:0.1, step:0.0001}
batch_size = 3 #@param {type:"slider", min:1, max:100, step:1}
pickle_dir = "processed/" #@param {type:"string"}
max_seq = 2048 #@param {type:"slider", min:1, max:3000, step:1}
epochs = 1416 #@param {type:"slider", min:1, max:10000, step:1}
model_save_path = "bin/models" #@param {type:"string"}
# Bound for the corpus harness, like train.py's --epochs 1 (ponder-lab/Python-Subjects#13).
epochs = 1

# %%
dataset = Data('dataset/processed/')
opt = Adam(par.l_r)
mt = MusicTransformer(
    embedding_dim=par.embedding_dim, 
    vocab_size=par.vocab_size, 
    num_layer=6, 
    max_seq=2048,
    debug=False)
mt.compile(optimizer=opt, loss=callback.TransformerLoss())

# %%
"""
# Train Model
"""

# %%
start_time = timeit.default_timer()
skipped_time = 0
for e in range(epochs):
    for b in range(len(dataset.files) // batch_size):
        batch_x, batch_y = dataset.seq2seq_batch(batch_size, par.max_seq)
        result_metrics = mt.train_on_batch(batch_x, batch_y)
        if b % 100 == 0:
          print_time = timeit.default_timer()
          print('===========================================\n')
          print(('Loss: {:6.6}, Accuracy: {:3.2}'.format(result_metrics[0], result_metrics[1])))
          mt.save('mt-2048-h4-dim256.h5', )
          skipped_time += timeit.default_timer() - print_time
          
        

time = timeit.default_timer() - start_time - skipped_time
write_csv(__file__, epochs=epochs, accuracy=float(result_metrics[1]), loss=float(result_metrics[0]), time=time)

# %%
"""
# Generate Model
"""

# %%
def fill_with_placeholder(prev_data: list, max_len: int, max_val: int=239):
    placeholder = [max_val for _ in range(max_len - len(prev_data))]
    return placeholder+prev_data

# %%
# Incomplete in the notebook (no loop body), so never executed there:
# for i in range(max_seq):
