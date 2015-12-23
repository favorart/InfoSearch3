#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import *
from operator import itemgetter
from itertools import groupby
from collections import defaultdict
import numpy as np
import pymorphy2
import itertools
import difflib
import codecs
import sys
import re
import os

from index_builder import StrictIndex, StrictIndexDocument


class SnippetBuilder(object):
    """ snippet_builder
            in   ( query & url/id of document )
            out  ( snippet                    )
    """
    def __init__(self, index, snippet_len=200):
        """ snippet_len - int
            index       - StrictIndex
        """
        self.snippet_len = snippet_len
        self.index = index

        self.docs  = self.index.read()
        if not self.docs: raise ValueError

        # 1. Полнота
        # 2. Слово-форма
        # 3. Компактность
        # 4. Порядок слов
        # 5. Неповторяемость слов
        # 6. Длина предложения от оптимального
        # 7. Близость к началу
        
        # self.weights = np.array([ 1., 1., 1., 1., 1., 1., 1. ])
        self.weights = np.array([ 10., 0.7, 0.5, 0.1, 0.4, 0.3, 0.1 ])

    def sentence_rank(self, sentence, query_words, in_sentence, len_text):
        """ norm: string, word: string, byte_pos: int

            sentence     - sentence[0] - begin:     int
                           sentence[1] - end:       int
                           sentence[2] - sentence:  string
            query_words  - normal words in query:   [(norm, word, no)] 
            in_sentence  - words in sentence (positions with repeats):
                           { (norm1, word1):[pos],
                             (norm2, word2):[pos0, pos1, ...],
                             ... }

            Returns the vector of sentence characteristics
                -->  maximaze to get the best
        """
        sent_cls = float(sentence[0]) / len_text # begin
        sent_len = fabs(self.snippet_len - (sentence[1] - sentence[0])) / self.snippet_len

        # print sentence[2][:20].encode('cp866', 'ignore')
        # print 'sent_len=', fabs(self.snippet_len - (sentence[1] - sentence[0])), self.snippet_len
        # print 'fullness=', len(in_sentence), len(query_words)
        
        fullness = 1. - float(len(in_sentence)) / len(query_words)
        # на самом деле в запросе могут быть тоже разные слово формы
        # n_norms_in_query = float(len(set([ norm for norm, word, pos  in query_words ])))
        # n_norms_in_sents = float(len(set([ norm for norm, word, poss in in_sentence ])))
        # fullness -= n_norms_in_sents / n_norms_in_query

        mw_forms = 0.
        # word form is equal?  --> min
        for norm, word, poss in in_sentence:
            for q_norm, q_word, w_pos in query_words:
                if q_norm == norm:
                    diffs = [ dif for dif in difflib.ndiff(q_word, word) if dif and (dif[0] == '+' or dif[0] == '-') ]
                    # print '\n'.join(diffs).encode('cp866', 'ignore')
                    # print 'diffs=', len(diffs), max(len(q_word), len(word))
                    mw_forms += len(diffs) / max(len(q_word), len(word))

        compacts = 0.
        poss = sorted([ pos for norm, word, poss in in_sentence for pos in poss ])
        for i in xrange(len(poss) - 1, 0, -1):
            compacts += poss[i] - poss[i-1]
        # print 'compacts=', compacts, (poss[-1] - poss[0] + 1)
        compacts /= (poss[-1] - poss[0] + 1)

        n_repeat = 1.
        for norm, word, poss in in_sentence:
            n_repeat *= len(poss)
        # print 'n_repeat=', n_repeat, len(poss)
        n_repeat /= len(poss)
        
        mw_order = 0.
        # itertools <-- TODO: !!!

        # 1. Полнота
        # 2. Слово-форма
        # 3. Компактность
        # 4. Неповторяемость слов
        # 5. Порядок слов
        # 6. Длина предложения от оптимального
        # 7. Близость к началу
        
        # --> min
        vector = ( 1. - fullness, 1. - mw_forms, 1. - compacts,
                   1. - n_repeat, 1. - mw_order,
                   1. - sent_len, 1. - sent_cls )
        # --> max
        return vector

    def document_rank(self, query_words, doc):
        """ query_words  - [(norm, word, byte_pos)]
            doc          - StrictIndexDocument()
        """
        # -----------------------------------------------
        found_sents, found_pos_s = {}, {}
        for (norm, word, pos) in query_words:
            found_pos_s[norm] = filter(lambda w: w[0] == norm, doc.words)

            found_sents[norm] = []
            for (norm1, word1, pos1) in found_pos_s[norm]:
                found_sents[norm] += filter(lambda s: s[0] < pos1 < s[1], doc.sentences)

        sentences = list(set([ s for lst in found_sents.values() for s in lst ]))
        poss =               [ w for lst in found_pos_s.values() for w in lst ]
        # -----------------------------------------------
        in_sentences = {}
        for s in sentences:
            in_sentences[s] = []
            for word, group in groupby(filter(lambda w: s[0] < w[2] < s[1], poss), itemgetter(1)):
                tpls = [ g for g in group ]
                in_sentences[s] += [ (tpls[0][0], word, [t[2] for t in tpls]) ]

        sents_ranks = {}
        for s, in_s in in_sentences.items():
            sents_ranks[s] = self.sentence_rank(s, query_words, in_s, doc.sentences[-1][1])
        # -----------------------------------------------
        # print '\n'
        # for s, rank in (sents_ranks.items()):
        #     print s[2][:20].encode('cp866', 'ignore'),
        #     for r in rank:
        #         print '%.3f' % r,
        #     print
        # print '\n'

        sent_values = []
        for s, rank in (sents_ranks.items()):
            sent_values.append( (s, np.sum(self.weights * rank)) )
            
            # print s[2][:20].encode('cp866', 'ignore'),
            # for w in self.weights * rank:
            #     print '%.3f' % w,
            # print

        sent_values = sorted(sent_values, key=itemgetter(1), reverse=True)
        # for s,pos in s_values:
        #     print (s[2][:20] + '\t%.3f' % pos).encode('cp866', 'ignore')
        # -----------------------------------------------
        return  self.snippet_format(sent_values, in_sentences)

    def snippet_format(self, s_values, in_sentences):
        """ Returns the formatted snippet """

        snippet_sent = s_values[0][0]
        # -----------------------------------------------
        # format snippet

        # too long
        if   len(snippet_sent[2]) > 1.3 * self.snippet_len:

            poss = [ pos for norm, word, poss in in_sentences[snippet_sent] for pos in poss ]
            poss.sort()

            rng = (poss[-1] - poss[0])
            if 1: # rng < self.snippet_len:

                bgn = poss[0] - snippet_sent[0]
                end = bgn + self.snippet_len

                for m in re.finditer(ur'[^а-яёА-ЯЁa-zA-Z0-9-]', snippet_sent[2][end:]):
                    snippet = u'... ' + snippet_sent[2][bgn : end + m.start()] + u' ...'
                    break
            else: # rng > self.snippet_len

                # !!! TODO !!!
                pass

        # too short
        elif len(snippet_sent[2]) < 0.7 * self.snippet_len:

            snippet_sents_len = 0
            snippet_sents = [ snippet_sent ]
            
            for more_sent, val in s_values[1:]:

                snippet_sents_len = sum(map(lambda s: len(s[2]), snippet_sents))
                if   0.7 * self.snippet_len > snippet_sents_len:
                    snippet_sents.append(more_sent)

                elif 1.3 * self.snippet_len < snippet_sents_len:
                    del snippet_sents[-1]

                else: # 0.7 * self.snippet_len < snippet_sents_len < 1.3 * self.snippet_len:

                    snippet_sents = sorted(snippet_sents, key=itemgetter(0))

                    snippet = u'... ' + snippet_sents[0][2]
                    for i in xrange(len(snippet_sents) - 1):
                        if (snippet_sents[i + 1][0] - snippet_sents[i][1]) < 5:
                            snippet += snippet_sents[i + 1][2]
                        else:
                            snippet += u' ... ' + snippet_sents[i + 1][2]
                    snippet += u' ...'
                    break

            else: # fail
                snippet = u'... ' + snippet_sent[2] + u' ...'

        # ideal
        else: snippet = '... ' + snippet_sent[2] + u' ...'
        # -----------------------------------------------
        return snippet

    def snippet(self, query, doc_id=-1, url=''):
        """ Returns the ready snippet """

        if url: doc = filter(lambda d: d.url == url, self.docs)[0]
        else:   doc = self.docs[doc_id]
        # -----------------------------------------------
        query_words = self.index.extract_words(query)
        snippet     = self.document_rank(query_words, doc)
        # -----------------------------------------------
        return snippet

