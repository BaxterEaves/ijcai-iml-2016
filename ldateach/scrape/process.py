# ldateach, generate/choose documents to teach topics model
# Copyright (C) 2016 Baxter S. Eaves Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


DOCSTR = """ Prepare scraped documents for LDA analysis."""

import pandas
import re
import os
import pickle
import string
from nltk.corpus import stopwords

TRANSTABLE = str.maketrans('', '', string.punctuation)
DIGITTABLE = str.maketrans('', '', string.digits)
STOP = set(stopwords.words('english'))
STOP.add('')
DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')


def process_csv(filename, filename_out, text_key='Text', min_word_ct=5):
    df = pandas.DataFrame.from_csv(filename)

    docs = []
    word_to_idx = {}
    idx_to_word = {}
    word_counts = {}
    words = set()

    print("- Calculating words and topics.")
    for idx in df.index:
        title = df.loc[idx]['Title']

        s = df.loc[idx][text_key]

        # XXX: Giffiths et al (2004) "...delimiting character, including
        # hyphens, was used to separate words, and we deleted any words that
        # occurred in less than five abstracts or belonged to a standard 'stop'
        # list used in computational linguistics, including numbers, individual
        # characters, and some function words."
        # I'm not sure how I feel about this method as it will surely remove
        # all reference to many proteins and chemical compounds, which are
        # doubtless important to the topic composition. On the other hand, how
        # does one discriminate between parethetical statements and lexically-
        # important delimiters (HINT: not with LDA)?

        # Remove all digits and split by non-alphanumeric characters.
        words_i = [w.lower() for w in re.split('\W', re.sub('\d', '', s))]
        # remove stop words and single-letter words
        words_i = [w for w in words_i if w not in STOP and len(w) > 1]
        words_u = list(set(words_i))  # gets unique words in document i

        doc = {
            'title': title,
            'words': words_i,
            'w': [],
            'counts': {},
            'id': idx}

        # Count the number of documents in which each word occurs
        for word in words_u:
            word_counts[word] = word_counts.get(word, 0) + 1

        words.update(words_u)
        docs.append(doc)

    # remove words that occur too few times
    remove_words = [w for w in word_counts if word_counts[w] < min_word_ct]
    for word in remove_words:
        words.remove(word)
        del word_counts[word]

    n_words = len(words)

    print("+ %d documents." % (len(docs),))
    print("+ %d words." % (n_words,))

    print("- Generating documents.")
    for widx, word in enumerate(words):
        word_to_idx[word] = widx
        idx_to_word[widx] = word

    words_in_docs = []
    for doc in docs:
        for word in doc['words']:
            widx = word_to_idx.get(word, None)
            if widx is not None:
                doc['w'].append(widx)
                doc['counts']['w'] = doc['counts'].get('w', 0) + 1
        words_in_docs.append(len(doc['w']))

    ave_words = sum(words_in_docs)/float(len(docs))
    print("+ Total number of words: %d" % (sum(words_in_docs,)))
    print("+ Average number of words/doc: %1.2f" % (ave_words,))
    print("- Saving.")
    output = {
        'docs': docs,
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'words': words,
        'word_counts': word_counts
    }
    pickle.dump(output, open(filename_out, 'wb'))
    print("- Saved as %s." % (filename_out,))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-f', '--filename', type=str)
    parser.add_argument('-k', '--text_key', type=str, default='Text',
                        help='Name of csv column with text to be processed.')
    parser.add_argument('-o', '--filename_out', type=str)
    parser.add_argument('-m', '--min_word_ct', type=int, default=5,
                        help='Minimum number of times a word must occur to '
                             'remain in the vocabulary.')

    kwargs = parser.parse_args()

    process_csv(kwargs.filename, kwargs.filename_out, text_key=kwargs.text_key,
                min_word_ct=kwargs.min_word_ct)
