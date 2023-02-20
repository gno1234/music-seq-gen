# music-seq-gen

Repository for personal study

This repository provides a step-by-step introduction to generating music. Tested only in Google Colab environment

The steps are below.

# Data preparation

## Step 1 : Public midi data set

Many grateful sources provide good quality midi data.

- maestro dataset (by google) 

[The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)

Huge data sets can be used for pre-training right away through a few steps.

If you use only public data set, go to step 3 

---

## Step 2 : Custom midi data set

### Step 2-1 : crop and conversion audio

If you have audio files such as .aac .mp3 .wav etc, you need to convert the audio format accordingly.

you can use `crop_audio.py`

```python
./data_prepare/crop_audio --audio_path ./your_path --method case_1
```

`./your_path` have to contain audio files.

there are three method to conversion audio files

`case_1` : just conversion audio format

`case_2` : Trim audio at fixed intervals

`case_3` : Trim audio at assigned position (crop_position.txt)

### Step 2-2 : From audio file to midi

We can also generate midi data from audio files

An audio file can be converted to a midi file through the model below.

- one set and offset (by google)
- music transcription (by google)

You can use the jupyter notebook `./data_prepare/Music_Transcription_with_Transformers_batch_processing.ipynb`

I modified original music_transcription jupyter notebook (by google) to batch processing

---

## Step 3 : Tokenize MIDI files and make dataset

MIDI datasets must be tokenized and prepared in an appropriate format before they can be input into the model.

We can use the midi tokenization library.

- MidiTok
[MidiTok](https://github.com/Natooz/MidiTok)

and you can use `midi_to_dataset.py`

```python
./data_prepare/midi_to_dataset.py --midi_path ./your_path
```

`./you_path` have to contain midi files. (subfolder ok)

unpacked `maestro-v3.0.0-midi.zip` file can be used directly 

output file is .npz format that contains **train_data** and **val_data**

---

## Step 4 : MIDI sequence generation model

We have to use sequence data generation model to generate midi music.

I changed **Attention** in GPT2 to **Relative Attention**.

Added relative attention to model based on karpathy/nanoGPT.

[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

Thank you so much for providing us with an easy-to-use GPT model.

Relative attention is based on:

[Relative Positional Encoding](https://jaketae.github.io/study/relative-positional-encoding/)

 

you can train the model using `train.py`

```python
train.py --config_yaml ./configyaml --npz_path ./your_npz_path
```

 you can adjust configuration of model and training parameter by modify config.yaml.

code is working with .npz file that made by `midi_to_dataset.py`

the checkpoint file will be saved in `./out`

---

## Step 5 : Generate MIDI sequence

When the model training is finished, you can create a MIDI sequence using `generate.py`

```python
generate.py  --max_new_tokens 1024 --context "False" --config_yaml config.yaml --weight_path ./out/ckpt.pt
```

You must use the same config.yaml you used for training.

If you want to use **context** to generate sequence, modify the `context.txt` and set `—context “True”`