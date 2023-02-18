from pathlib import Path
import argparse
import glob
import os
import random
import numpy as np
import json
from tqdm import tqdm

from miditok import  REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile


parser = argparse.ArgumentParser(description='crop and save audio file')

parser.add_argument("--midi_path", type = str)
parser.add_argument("--split_ratio", type = float, default = 0.85)
parser.add_argument("--padding_size", type = float, default = 512)

args = parser.parse_args()




midi_path = args.midi_path

pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}  # (min, max)

tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens)

midi_list = glob.glob(os.path.join(midi_path,'**/*.midi'), recursive=True)
midi_list.extend(glob.glob(os.path.join(midi_path,'**/*.mid'), recursive=True))
print(midi_list)

print("number of midi files :", len(midi_list))

random.shuffle(midi_list)

split_ratio = args.split_ratio

n = int(len(midi_list)*split_ratio)
midi_list_train = midi_list[:n]
midi_list_val = midi_list[n:]

print("number of train_midi",len(midi_list_train))
print("number of val_midi",len(midi_list_val))

#train agumentation
data_augmentation_offsets = [2, 2, 1]  # perform data augmentation on 2 pitch octaves, 2 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_list_train, Path('./aug_split/train/'), None, data_augmentation_offsets)


padding_size = args.padding_size

json_list = glob.glob("./aug_split/train/*.json")
random.shuffle(json_list)
zero_padding = [0 for i in range(512)]

t_train = []
t_train.extend(zero_padding)
for file_path in tqdm(json_list):
    with open(file_path, "r") as json_file:
        j = json.load(json_file)
    t_train.extend(j["tokens"][0])

train_data = np.array(t_train)



#val agumentation
data_augmentation_offsets = [2, 2, 1]  # perform data augmentation on 2 pitch octaves, 2 velocity and 1 duration values
tokenizer.tokenize_midi_dataset(midi_list_val, Path('./aug_split/val/'), None, data_augmentation_offsets)

json_list = glob.glob("/content/aug_split/val/*.json")
random.shuffle(json_list)
zero_padding = [0 for i in range(512)]

t_val = []
t_val.extend(zero_padding)
for file_path in tqdm(json_list):
    with open(file_path, "r") as json_file:
        j = json.load(json_file)
    t_val.extend(j["tokens"][0])

val_data = np.array(t_val)

np.savez("./aug_tokenized_midi_data", train_data=train_data, val_data=val_data)