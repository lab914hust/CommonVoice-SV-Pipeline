import pickle
import argparse
import csv
import numpy as np
from scipy import spatial

def similar_speaker_removal(emb_path, threshold):
    with open(emb_path, 'rb') as handle:
        data = pickle.load(handle)

    print("----------COMPUTING SIMILARITY SCORES----------")
    all_spk_mean_emb = []
    duplicate_spk = {}
    for spk, utt_list in data.items():
        spk_embs = []
        for i in range(len(utt_list)):
            spk_embs.append(np.squeeze(utt_list[i]['embedding']) / np.linalg.norm(np.squeeze(utt_list[i]['embedding']), ord=2))
        all_spk_mean_emb.append((spk, np.sum(np.array(spk_embs), axis=0) / len(spk_embs)))
    for i in range(len(all_spk_mean_emb)):
        for j in range(i+1, len(all_spk_mean_emb) - 1):
            cosine = 1 - spatial.distance.cosine(all_spk_mean_emb[i][-1], all_spk_mean_emb[j][-1])
            if cosine >= threshold:
                duplicate_spk[(all_spk_mean_emb[i][0], all_spk_mean_emb[j][0])] = float(cosine)
    return duplicate_spk

def save_to_csv(spk_avg_score, save_path):
    with open(save_path, 'w') as target:
        writer = csv.writer(target, delimiter='\t')

        print("----------SAVING SCORES TO CSV----------")

        writer.writerow(['spk_id1', 'spk_id2', 'avg_score'])
        for spk, avg_score in spk_avg_score.items():
            writer.writerow([spk[0], spk[1], avg_score])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str)
    parser.add_argument('--threshold', type=float, default=0.85)
    args = parser.parse_args()

    save_path = args.emb_path.split('/')[-1].split('.')[0] + '_similar_spk.csv'
    duplicate_spk = similar_speaker_removal(args.emb_path, args.threshold)
    save_to_csv(duplicate_spk, save_path)


