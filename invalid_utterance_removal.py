import pickle
import argparse
import csv
from sklearn.metrics import pairwise_distances
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def invalid_speaker_removal(emb_path, threshold):
    with open(emb_path, 'rb') as handle:
        data = pickle.load(handle)

    print("----------FINDING INVALID UTTERANCES----------")
    spk_invalid_utt = {k:[] for k in list(data.keys())}
    for spk, utt_list in data.items():
        score_matrix = np.array([np.squeeze(i['embedding']) / np.linalg.norm(np.squeeze(i['embedding']), ord=2) for i in utt_list])
        cos_matrix = 1 - pairwise_distances(score_matrix, metric="cosine")
        if cos_matrix.shape[0] == cos_matrix.shape[1] and cos_matrix.shape[0] == len(utt_list):
            invalid_index = []

            q1 = np.quantile(cos_matrix, q=0.25)
            q3 = np.quantile(cos_matrix, q=0.75)
            
            iqr = q3 - q1
            a_min = q1 - threshold * iqr
            a_max = q3 + threshold * iqr

            for i in range(cos_matrix.shape[0]):
                a_i = (np.sum(cos_matrix[i, :]) - 1) / (cos_matrix.shape[0] - 1)
                if a_i > a_max or a_i < a_min:
                    invalid_index.append(i)
            for i in invalid_index:
                spk_invalid_utt[spk].append(data[spk][i]['audio'])
    return spk_invalid_utt

def save_to_csv(spk_invalid_utt, save_path):
    with open(save_path, 'w') as target:
        writer = csv.writer(target, delimiter='\t')

        print("----------SAVING TO CSV----------")

        writer.writerow(['spk_id', 'invalid_audio'])
        for spk, invalid_utt_list in spk_invalid_utt.items():
            for i in invalid_utt_list:
                writer.writerow([spk, i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str)
    parser.add_argument('--thresh_hold', type=float, default=1.2)
    args = parser.parse_args()

    save_path = args.emb_path.split('/')[-1].split('.')[0] + '_invalid_utt.csv'
    spk_invalid_utt = invalid_speaker_removal(args.emb_path, args.thresh_hold)
    save_to_csv(spk_invalid_utt, save_path)


