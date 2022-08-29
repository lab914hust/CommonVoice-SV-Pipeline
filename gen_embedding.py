import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pickle
import pandas as pd
import argparse
from tqdm import tqdm

def gen_embedding(device, csv_path, clips_path, save_path):

    classifier = EncoderClassifier.from_hparams(run_opts={"device":device}, source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

    embeddings = {}
    data_csv = pd.read_csv(csv_path, delimiter='\t')
    dir_pkl = save_path

    embeddings = {k:[] for k in set(data_csv['client_id'].to_list())}

    print("----------GENERATING EMBEDDINGS----------")
    for i in tqdm(range(data_csv.shape[0])):
        try:
            audio_path = os.path.join(clips_path, data_csv['path'][i].replace('.mp3', '.wav'))
            if os.path.exists(audio_path):            
                signal, _ =torchaudio.load(audio_path)
                embedding = classifier.encode_batch(signal.to(device))
                embeddings[data_csv['client_id'][i]].append({'audio': audio_path,  'embedding': embedding.detach().cpu().numpy()})
        except:
            continue

    with open(dir_pkl, 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--clips_path', type=str)
    args = parser.parse_args()

    save_path = args.clips_path.split('/')[-2] + '.pkl'
    gen_embedding('cuda', args.csv_path, args.clips_path, save_path)


