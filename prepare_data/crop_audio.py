import shutil
import os
import glob
from tqdm import tqdm
from pydub import AudioSegment
import argparse

parser = argparse.ArgumentParser(description='crop and save audio file')

parser.add_argument("--audio_path", type = str)
parser.add_argument("--crop_position_txt", type = str, default = "./crop_position.txt")
parser.add_argument("--crop_sec", type = int, default = 30, help ="sec unit")
parser.add_argument("--sample_rate", type = int, default = 16000, help ="sec unit")
parser.add_argument("--audio_format", type =str, default = "wav" )
parser.add_argument("--method", type = str)
parser.add_argument("--out_path", type = str, default = './out')

args = parser.parse_args()

print(args.audio_path)


#setting
audio_path = args.audio_path
crop_position_txt = args.crop_position_txt
crop_sec = args.crop_sec
sample_rate = args.sample_rate
audio_format = args.audio_format
out_path = args.out_path

audio_files = glob.glob(audio_path + "/*")
os.makedirs(out_path, exist_ok=True)

# variables : audio_path, crop_position_list, crop_sec, sample_rate, audio_format, crop_position_list

#audio file format to .wav
if args.method == 'case_1':
    n = 0
    for file in tqdm(audio_files):
        m = AudioSegment.from_file(file)
        m = m.set_channels(1)
        m = m.set_frame_rate(sample_rate)
        title = "audio_"+ str(n).zfill(5)
        out = os.path.join(out_path,title)+".wav"
        m.export(out, format="wav")
        n += 1

#crop fixed interval for long audio file
elif args.method == 'case_2':
    n = 0
    for file in tqdm(audio_files):

        music = AudioSegment.from_file(file)
        length = music.duration_seconds
        title = "audio_"+ str(n).zfill(3)
        format_ = "." + audio_format
        num_crop = length/crop_sec
        n += 1

        i = 0
        while i < num_crop-1:
        
            croped_music = music[i*crop_sec*1000:(i+1)*crop_sec*1000]
            croped_music = croped_music.set_channels(1)
            croped_music = croped_music.set_frame_rate(sample_rate)
            file_index = str(i).zfill(4)
            file_name = title + "_" + file_index + format_
            file_path = os.path.join(out_path, file_name)
            croped_music.export(file_path, format="wav")
            i += 1

#assign crop position

elif args.method == 'case_3':
    print(args.crop_position_txt)
    with open(crop_position_txt) as txt:
        crop_position_list = txt.read().split(',')
        crop_position_list = list(map(int, crop_position_list))

    music = AudioSegment.from_file(audio_files[0])
    start = 0
    i = 0
    for t in tqdm(crop_position_list):
        music = music.set_channels(1)
        music = music.set_frame_rate(16000)
        end = t
        croped_music = music[start*1000:end*1000]
        start = t

        file_index = str(i).zfill(3)
        file_name = "audio_" + file_index + ".wav"
        file_path = os.path.join(out_path, file_name)
        i += 1

        croped_music.export(file_path, format="wav")



