import datetime

from absl import logging

import numpy as np
import pandas as pd


def get_cosine_similarity(sents, embed):

    # Reduce logging output.
    logging.set_verbosity(logging.ERROR)
    # with tf.Session() as session:
    #     session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    embeddings = embed(sents)

    # this is the cosine score but since these embeddings are normalised
    # we can just take the inner product (similar to dot product for 1D arrays)
    # this gives us a score of how closely related the sentences are.
    return np.inner(embeddings[0], embeddings[1])
