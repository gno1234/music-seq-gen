
import os
import yaml
import argparse
import torch

from model import GPTConfig, GPT

from miditok import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile

parser = argparse.ArgumentParser()
parser.add_argument("--config_yaml", type = str, default = "./config.yaml")
parser.add_argument("--weight_path", type = str, default = "./out/ckpt.pt")
parser.add_argument("--max_new_tokens",type = int ,default = 1024)
parser.add_argument("--context", type = str, default = "False" )
args = parser.parse_args()

with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

exp_title = config['exp_title']

n_layer = config['n_layer']
n_head = config['n_head']
n_embd = config['n_embd']

block_size = config['block_size']
vocab_size = config['vocab_size']

dropout = float(config['dropout'])

dtype = config['dtype']
bias = False

weight_path = args.weight_path

device ="cuda:0"

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)


print(f"load model weight from {weight_path}")

ckpt_path = weight_path
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]

# create the model##########################################
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']

model.eval()
model.to(device)
############################################################

pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens)

# context load ############################################
if args.context == "True":
    print("context is",args.context)
    with open("./context.txt") as txt:
        uploaded_tokens = txt.read().split(',')
        uploaded_tokens = list(map(int, uploaded_tokens))
elif args.context == "False":
    uploaded_tokens = [1]
############################################################

context = torch.tensor([uploaded_tokens], device='cuda:0')

max_new_tokens = args.max_new_tokens

generated_tokens =  model.generate(context, max_new_tokens=max_new_tokens, temperature = 1)[0].tolist()
converted_back_midi = tokenizer([generated_tokens]) #get_midi_programs(midi)

os.makedirs("./midi_export", exist_ok=True)
midi_export_path ="./midi_export/"+ exp_title + ".mid"
converted_back_midi.dump(midi_export_path)
print("midi_export_path")