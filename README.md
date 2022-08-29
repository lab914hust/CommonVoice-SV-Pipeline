
# CommonVoice Speaker Verification Pipeline For Pre-processing

## Dependencies

Requirements:
```
pip install -r requirements.txt
```

## Convert MP3 to WAV

Each data file downloaded from CommonVoice will have a clips folder containing MP3 files. We first convert them to WAV:
```
python convert_to_wav.py --clips_path=<path to 'clips' folder>
```

## Generate Speaker Embeddings

After converting audio to WAV, generate the speaker embeddings of the data using the validated list downloadeds:
```
python gen_embedding.py --csv_path=<path to validated.tsv file> --clips_path=<path to 'clips' folder>
```

## Data Pre-processing

First we remove the invalid speakers from the data. Every speaker whose average similarity score among his/her utterances lower than the threshold (default 0.4) will be removed:
```
python invalid_speaker_removal.py --emb_path=<path to pkl file> --threshold=<removing threshold. default = 0.4>
```
After removing invalid speakers, we remove the invalid utterances:
```
python invalid_utterance_removal.py --emb_path=<path to pkl file>
```
Now we find the pairs of speaker which are similar to each other. Every pair of speakers with average similarity score higher than the threshold (default 0.85) will be considered one speaker:
```
python similar_speaker_removal.py --emb_path=<path to pkl file> --threshold=<similar threshold. default = 0.85>
```
Finally, prepair the final CSV for training:
```
python finalize.py --emb_path=<path to pkl file>
```
