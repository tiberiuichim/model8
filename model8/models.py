from .config import MAX_NB_WORDS, DATA_DIR, TOKENIZER_FILENAME
from .config import MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT
from .config import METADATA_FILENAME, STATS_FILENAME, MODEL_FILENAME
from .core import make_tokenizer, split_data, make_model
from .utils import folder_lock
from keras.models import load_model
from nltk.tokenize.punkt import PunktSentenceTokenizer
from repoze.lru import CacheMaker
from six.moves import cPickle
from sqlalchemy import Column, Index, Integer, Text, String
from sqlalchemy import ForeignKey, Boolean
from sqlalchemy import engine_from_config
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import configure_mappers
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData
import json
import logging
import numpy as np
import os.path
import zope.sqlalchemy

logger = logging.getLogger('model8')
cache = CacheMaker(maxsize=10)


# Recommended naming convention used by Alembic, as various different database
# providers will autogenerate vastly different names making migrations more
# difficult. See: http://alembic.zzzcomputing.com/en/latest/naming.html
NAMING_CONVENTION = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)
Base = declarative_base(metadata=metadata)


class MLModel(Base):
    """ The Machine Learning Model, represented on disk and in the database.

    To build a model that grants prediction capabilities, the following
    process is needed:

        - Gather all texts and split them in sentences. Each sequence is
        labeled with a category.
        - Build a word tokenizer from all the sentences. This is a two step
        process.  First, the tokenizer is fitted with the texts. This builds
        a corpus of words and assigns a numeric index to all words, where
        position 1 is the most common word and increments to the biggest
        number, for the least common words. Second, the text data is
        transformed into numeric sequences by the tokenizer, using already
        computed indexes for each word.
        - Split the sequences spit by the tokenizer in two sets: one for
        training and one for testing. They should always be kept separate.
        - Assemble the neural network.
        - Train the neural network with the training/testing data.
        - Save the model and the tokenizer to the disk.
        - On startup, the server needs to load all models to memory and uses
        them to make predictions.

    """
    __tablename__ = 'mlmodel'

    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)
    # model_data_path = Column(String(255))

    fragments = relationship("DataFragment")

    def _iter_text_data(self):
        pst = PunktSentenceTokenizer()
        for fragment in self.fragments:
            text = (fragment.text or '').strip()
            if not text:
                continue
            label = fragment.label
            sentences = pst.sentences_from_text(fragment.text)
            for sentence in sentences:
                # yield sentence.encode('utf-8'), label
                yield sentence, label   # py3

    # @lru_cache(2)
    def build_tokenizer(self, max_nb_words=MAX_NB_WORDS):
        """ Prepares the data, builds the tokenizer, saves it to disk
        """

        # sentences is a list of strings
        # _labels is a list of words (labels). Each coresponds to an item
        # in sentences, labeling that sentence
        sentences, _labels = zip(*self._iter_text_data())

        # we want to assign numeric values to labels, while also building a
        # mapping of the original labels

        label_index = {}
        labels = []
        for l in _labels:
            if l not in label_index:
                label_index[l] = len(label_index)
            labels.append(label_index[l])

        tokenizer = make_tokenizer(sentences, max_nb_words)

        statistics = {
            'sentences': len(labels),
            'nb_words': len(tokenizer.word_index)
        }

        with folder_lock(self.datadir):
            with open(self._tokenizer_path, 'wb') as f:
                cPickle.dump(tokenizer, f)

            with open(self._stats_path, 'w') as f:
                json.dump(statistics, f)

        # now split the texts into training and test data
        X_train, y_train, X_test, y_test, labels = split_data(
            tokenizer, sentences, labels, max_seq_len=MAX_SEQUENCE_LENGTH,
            vsplit=VALIDATION_SPLIT
        )

        metadata = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'labels': label_index,
        }
        with folder_lock(self.datadir):
            with open(self._metadata_path, 'wb') as f:
                cPickle.dump(metadata, f)

        return tokenizer, X_train, y_train, X_test, y_test, labels

    def labels(self):
        """ Returns a mapping label_index: label_title
        """
        with open(self._metadata_path, 'rb') as f:
            meta = cPickle.load(f)
        return meta['labels']

    # @lru_cache(2)
    def build(self):
        if self._is_locked():
            raise ValueError("Data is locked")

        tokenizer, X_train, y_train, X_test, y_test, labels = \
            self.build_tokenizer(max_nb_words=MAX_NB_WORDS)
        model = make_model(X_train, y_train, X_test, y_test,
                           max_nb_words=MAX_NB_WORDS)
        with folder_lock(self.datadir):
            model.save(self._model_path)
        return model

    def predict(self, sentence):
        # because some sentences can contain words that don't exist in the
        # tokenizer, the model we split
        # Note: Assumes one sentence only
        sentence = sentence.decode('utf-8')     # ???
        tokens = self.tokenizer.texts_to_sequences([sentence])
        sequence = np.array(tokens)
        if len(sequence[0, ]) == 0:
            logger.info("Cannot predict, no tokens recognized")
            return {}
        model = self.get_keras_model()
        res = model.predict_classes(sequence)
        labels = self.labels()
        labels = {v: k for k, v in labels.items()}  # invertify dict
        return labels[res[0, 0]]

    def statistics(self):
        sp = self._stats_path
        if self._is_locked() or (not os.path.exists(sp)):
            raise ValueError

        with open(sp) as f:
            stats = json.load(f)

        return stats

    # @lru_cache(2)
    @property
    def tokenizer(self):
        """ Returns the tokenizer object
        """
        tp = self._tokenizer_path
        if self._is_locked() or (not os.path.exists(tp)):
            raise ValueError

        with open(tp, 'rb') as f:
            tokenizer = cPickle.load(f)

        return tokenizer

    @cache.lrucache()
    def get_keras_model(self):
        """ Returns the model object
        """
        mp = self._model_path
        if self._is_locked() or (not os.path.exists(mp)):
            raise ValueError
        print("=" * 20, 'loading model')
        return load_model(mp)

    @property
    def datadir(self):
        return os.path.join(DATA_DIR, self.name)

    @property
    def _tokenizer_path(self):
        return os.path.join(self.datadir, TOKENIZER_FILENAME)

    @property
    def _model_path(self):
        return os.path.join(self.datadir, MODEL_FILENAME)

    @property
    def _metadata_path(self):
        return os.path.join(self.datadir, METADATA_FILENAME)

    @property
    def _stats_path(self):
        return os.path.join(self.datadir, STATS_FILENAME)

    def _is_locked(self):
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        return '.lock' in os.listdir(self.datadir)

    def can_predict(self):
        """ Make a guess if the model is ready to make predictions
        """
        if self._is_locked():
            return False

        for fn in [self._tokenizer_path,
                   self._model_path,
                   self._metadata_path,
                   self._stats_path]:
            if not os.path.exists(fn):
                return False

        return True

    def reset(self):
        """ Removes all model associated files. All for a fresh start.
        """
        if self._is_locked():
            return ValueError("Cannot reset while locked")

        for fn in [self._tokenizer_path,
                   self._model_path,
                   self._metadata_path,
                   self._stats_path]:
            if os.path.exists(fn):
                os.remove(fn)


class DataFragment(Base):
    __tablename__ = 'datafragment'

    id = Column(Integer, primary_key=True)
    text = Column(Text)
    label = Column(String(255), nullable=False)
    # Boolean needs name, see
    # https://bitbucket.org/zzzeek/sqlalchemy/issues/3067/naming-convention-exception-for-boolean
    processed = Column(Boolean(name='processed'), default=False)

    mlmodel_id = Column(Integer, ForeignKey("mlmodel.id"))

Index('mlmodel_id', DataFragment.mlmodel_id)
# TODO: add label index


# import or define all models here to ensure they are attached to the
# Base.metadata prior to any initialization routines
# run configure_mappers after defining all of the models to ensure
# all relationships can be setup
configure_mappers()


def get_engine(settings, prefix='sqlalchemy.'):
    return engine_from_config(settings, prefix)


def get_session_factory(engine):
    factory = sessionmaker()
    factory.configure(bind=engine)
    return factory


def get_tm_session(session_factory, transaction_manager):
    """
    Get a ``sqlalchemy.orm.Session`` instance backed by a transaction.

    This function will hook the session to the transaction manager which
    will take care of committing any changes.

    - When using pyramid_tm it will automatically be committed or aborted
      depending on whether an exception is raised.

    - When using scripts you should wrap the session in a manager yourself.
      For example::

          import transaction

          engine = get_engine(settings)
          session_factory = get_session_factory(engine)
          with transaction.manager:
              dbsession = get_tm_session(session_factory, transaction.manager)

    """
    dbsession = session_factory()
    zope.sqlalchemy.register(
        dbsession, transaction_manager=transaction_manager)
    return dbsession


def includeme(config):
    """
    Initialize the model for a Pyramid app.

    Activate this setup using ``config.include('model8.models')``.

    """
    settings = config.get_settings()

    # use pyramid_tm to hook the transaction lifecycle to the request
    config.include('pyramid_tm')

    session_factory = get_session_factory(get_engine(settings))
    config.registry['dbsession_factory'] = session_factory

    # make request.dbsession available for use in Pyramid
    config.add_request_method(
        # r.tm is the transaction manager used by pyramid_tm
        lambda r: get_tm_session(session_factory, r.tm),
        'dbsession',
        reify=True
    )
