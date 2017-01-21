DATA_DIR = './data'

MODEL_FILENAME = 'model.hdf5'
TOKENIZER_FILENAME = 'tokenizer.pickle'
METADATA_FILENAME = 'metadata.pickle'
STATS_FILENAME = 'statistics.json'

VALIDATION_SPLIT = 0.2      # ratio of test/training data split
MAX_SEQUENCE_LENGTH = 20    # sentences will be cut/padded to this size
MAX_NB_WORDS = 15000        # max number of words to tokenize
