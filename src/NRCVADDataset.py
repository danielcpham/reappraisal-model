from datasets import Dataset, load_dataset

data = load_dataset('text', data_files={
    'valence': '/Users/danielpham/Downloads/NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/v-scores.txt',
    'arousal': '/Users/danielpham/Downloads/NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/a-scores.txt',
    'dominance': '/Users/danielpham/Downloads/NRC-VAD-Lexicon-Aug2018Release/OneFilePerDimension/d-scores.txt'

})
