import pandas as pd
import argparse
import csv
import pickle

def main(name, emb_path, save_path):

    with open(emb_path, 'rb') as handle:
        data = pickle.load(handle)
    data_dict = {'client_id': [], 'path': []}
    for spk, utt_list in data.items():
        for utt in utt_list:
            data_dict['client_id'].append(spk)
            data_dict['path'].append(utt['audio'])
    data_all = pd.DataFrame(data=data_dict)

    invalid_speaker = pd.read_csv(name + '_invalid_spk.csv', delimiter='\t')
    invalid_utt = pd.read_csv(name + '_invalid_utt.csv', delimiter='\t')
    similar_spk = pd.read_csv(name + '_similar_spk.csv', delimiter='\t')

    print('-------------------')
    print('NUMBER OF SPEAKERS:', len(set(data_all['client_id'].to_list())))
    print('NUMBER OF UTTERANCES:', data_all.shape[0])

    with open(save_path, 'w') as target:
        writer = csv.writer(target, delimiter='\t')
        writer.writerow(['path', 'spk_id'])
        for spk in invalid_speaker.spk_id.to_list():
            data_all.drop(data_all.index[data_all['client_id'] == spk], inplace=True)
        for i in similar_spk.index:
            data_all.loc[data_all["client_id"] == similar_spk['spk_id2'][i], "client_id"] = similar_spk['spk_id1'][i]
        for i in invalid_utt.index:
            data_all.drop(data_all.index[data_all['path'] == invalid_utt['invalid_audio'][i]], inplace=True)
        for i in data_all.index:
            writer.writerow([data_all['path'][i], data_all['client_id'][i]])

    print('-------------------')
    print('NUMBER OF SPEAKERS LEFT:', len(set(data_all['client_id'].to_list())))
    print('NUMBER OF UTTERANCES LEFT:', data_all.shape[0])
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str)
    args = parser.parse_args()

    name = args.emb_path.split('/')[-1].split('.')[0]
    save_path = name + '_final.csv'
    main(name, args.emb_path, save_path)