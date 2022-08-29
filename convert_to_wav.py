import argparse
import os
import glob
from pydub import AudioSegment
import librosa
import soundfile as sf
from tqdm import tqdm

def convert_to_wav(dir):
    print("----------CONVERTING MP3 TO WAV----------")
    for i in tqdm(glob.glob(dir + '/*.mp3')):
        try:
            audio = AudioSegment.from_mp3(i)
            audio.export(os.path.splitext(i)[0] + '.wav', format="wav")
            os.remove(i)
        except:
            continue

def resample_wav(dir):
    print("----------RESAMPLING WAV----------")
    for i in tqdm(glob.glob(dir + '/*.wav')):
        try:
            audio, _ = librosa.load(i, sr=16000)
            sf.write(i, audio, 16000)
        except:
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clips_path', type=str)

    args = parser.parse_args()
    convert_to_wav(args.clips_path)
    resample_wav(args.clips_path)