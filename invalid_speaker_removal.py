import pickle
import argparse
import csv
from sklearn.metrics import pairwise_distances
import numpy as np

def invalid_speaker_removal(emb_path):
    with open(emb_path, 'rb') as handle:
        data = pickle.load(handle)

    print("----------COMPUTING SIMILARITY SCORES----------")
    spk_avg_score = {}
    for spk, utt_list in data.items():
        score_matrix = np.array([np.squeeze(i['embedding']) / np.linalg.norm(np.squeeze(i['embedding']), ord=2) for i in utt_list])
        cos_matrix = 1 - pairwise_distances(score_matrix, metric="cosine")
        if cos_matrix.shape[0] == cos_matrix.shape[1] and cos_matrix.shape[0] == len(utt_list):
            spk_avg_score[spk] = np.mean(cos_matrix)
    return spk_avg_score

def save_to_csv(spk_avg_score, save_path, threshold):
    with open(save_path, 'w') as target:
        writer = csv.writer(target, delimiter='\t')

        print("----------SAVING SCORES TO CSV----------")

        writer.writerow(['spk_id', 'avg_score'])
        for spk, avg_score in spk_avg_score.items():
            if float(avg_score) < threshold:
                writer.writerow([spk, avg_score])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str)
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()

    save_path = args.emb_path.split('/')[-1].split('.')[0] + '_invalid_spk.csv'
    spk_avg_score = invalid_speaker_removal(args.emb_path)
    save_to_csv(spk_avg_score, save_path, args.threshold)


