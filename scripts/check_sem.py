import pickle
for fname in ['semantic_train.pkl', 'semantic_validation.pkl', 'semantic_test.pkl',
              'embeddings_train.pkl', 'embeddings_validation.pkl', 'embeddings_test.pkl']:
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    keys = list(d.keys())
    entry = d[keys[0]]
    vec = entry['vector']
    print(f'{fname}: {len(d)} entries, key_type={type(keys[0]).__name__}, '
          f'vec_dim={len(vec)}, label={entry["label"]}')
