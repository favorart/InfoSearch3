#!/usr/bin/env python
# -*- coding: utf-8 -*-

from boilerpipe.extract import Extractor
import pymorphy2
import codecs
import sys
import re
import os

from sentence_splitter import SentenceSplitter


class StrictIndexDocument(object):
    """ """
    def __init__ (self):
        """ """
        self.id = 0
        self.url = ''
        self.words = []      # (word,norm,position)
        self.sentences = []  # (begin,end,sentence)
        self.n_sentences = 0
        self.n_words = 0

    def __str__  (self):
        return (u'id=%d\t%s\tn_words=%d\n%s\n\nn_sentences=%d\n%s\n\n' % \
                (self.id, self.url, self.n_words, \
                 u'\n'.join([ '%s\t%s\t%d'      % (w[0], w[1], w[2]) for w in self.words ]), \
                 self.n_sentences, 
                 u'\n'.join([ '%d\t%d:\t\'%s\'' % (s[0], s[1], s[2]) for s in self.sentences ])
                 )).encode('cp866', 'ignore')


class StrictIndex(object):
    """ index_builder 
            in   набор документов 
            out  прямой индекс
    """
    def __init__(self, fn_index, SS=SentenceSplitter()):
        """ """
        self.morph = pymorphy2.MorphAnalyzer()
        self.fn_index = fn_index

        self.re_extract_words = re.compile(ur'[^a-zа-яё0-9-]')
        self.re_repeat_spaces = re.compile(ur'[ ]+')
        self.re_set_paragraph = re.compile(ur'\r?\n')

        self.SS = SS
        # ---------------------------------
        if not os.path.exists(u'data'):
            os.makedirs(u'data')

        self.SS.fit(fn_fit_corpus)
        # ---------------------------------
        self.word_min_len = 3
        self.threshold_extractor_fails = 0.1

    def extract_words(self, text):
        """ """
        doc_words = []

        word_flag = False
        poss = []
        words = self.re_extract_words.sub(u' ', text.lower())
        
        for i,c in enumerate(words):
            if not word_flag and (c != ' '):
                poss.append(i)
                word_flag = True
            elif   word_flag and (c == ' '):
                word_flag = False

        words = self.re_repeat_spaces.sub(u' ', words)
        words = re.sub(ur'(?:^[ ]+)|(?:[  ]$)', u'', words)
        words = re.split(ur'[ ]', words)

        # print '\n'.join(words[:10]).encode('cp866', 'ignore')        
        # with codecs.open(u'outs111.txt', 'a', encoding='utf-8') as f_out:
        #     for pos, word in zip(poss, words):
        #         print >>f_out, pos, word

        for pos, word in zip(poss, words):
            if len(word) >= self.word_min_len:
                norm = self.morph.parse(word)[0].normal_form
                # print '  '.join([norm, word, str(pos)]).encode('cp866', 'ignore')
                doc_words.append( (norm, word, pos) )
        return doc_words

    def build(self, urls, docs=None, fn_fit_corpus=u'data/sentences.xml', extacted_cache=False):
        """ """
        indexed_docs = []
        doc_id = 0
        # ---------------------------------
        if docs and len(urls) != len(docs):
            raise ValueError

        for i,URL in enumerate(urls):
            doc = StrictIndexDocument()
            doc.id, doc.url = doc_id, URL
            doc_id += 1

            if os.path.exists('./data/extracted-'+ str(doc_id) +'-cached.txt'):
                with codecs.open('./data/extracted-'+ str(doc_id) +'-cached.txt', 'r', encoding='utf-8') as f:
                    extr_text = f.read()
            else:
                if docs:
                    extractor = Extractor(extractor='ArticleExtractor', html=docs[i])
                else:
                    extractor = Extractor(extractor='ArticleExtractor', url=URL)
                extr_text = extractor.getText()
                # ---------------------------------
                if len(extr_text) < len(docs[i]) * self.threshold_extractor_fails:
                    if docs:
                        exKeepAll = Extractor(extractor='KeepEverythingExtractor', html=docs[i])
                    else:
                        exKeepAll = Extractor(extractor='KeepEverythingExtractor', url=URL)
                    extr_text = exKeepAll.getText()
                # ---------------------------------
                if extacted_cache:
                    with codecs.open('./data/extracted-'+ str(doc_id) +'-cached.txt', 'w', encoding='utf-8') as f:
                        print >>f, extr_text

            doc.words = self.extract_words(extr_text)
            doc.n_words = len(doc.words)

            paragraph = self.re_set_paragraph.sub(u' ', extr_text)
            sentences, poss = self.SS.predict(paragraph)

            # with codecs.open(u'data/outs.txt', 'a',encoding='utf-8') as f_out:
            #      print >>f_out, u'\n\n-----\n', u','.join( [ str(p) for p in poss ] )
            #      # print >>f_out, u','.join( [ str(p) for p in poss2 ] )
            #      for s in sentences:
            #          print >>f_out, s, '\n'

            poss = [0] + list(poss)
            doc.sentences = [ (poss[i], poss[i+1], sentences[i]) for i in xrange(len(poss)-1) ]
            doc.n_sentences = len(doc.sentences)

            indexed_docs.append(doc)
        return indexed_docs
    
    def dump(self, docs):
        """ """
        # sys.stdout = codecs.getwriter('cp866')(sys.stdout)
        with codecs.open(self.fn_index, 'w', encoding='utf-8') as f_index:
            for doc in docs:
                print >>f_index, u'%d\t%s\t%d\t%d\n' % (doc.id, doc.url, doc.n_words, doc.n_sentences)

                for tuple in doc.words:
                    # tuple: (word,norm,position)
                    print >>f_index, u'%s\t%s\t%d' % (tuple[0], tuple[1], tuple[2])

                for sentence in doc.sentences:
                    # sentence: (begin,end,sentence)
                    print >>f_index, u'%d\t%d\t%s' % (sentence[0], sentence[1], sentence[2])

                print >>f_index, '\n\n'

    def read(self):
        """ """
        docs = []
        n, m = 0, 0
        with codecs.open(self.fn_index, 'r', encoding='utf-8') as f_index:
            for line in f_index.readlines():
                splt = line.strip().split(u'\t')

                if not (n or m) and len(splt) == 4:
                    doc = StrictIndexDocument()
                    # header
                    doc.id, doc.url = int(splt[0]), splt[1]
                    n = doc.n_words = int(splt[2])
                    m = doc.n_sentences = int(splt[3])
                    docs.append(doc)

                elif      n > 0 and len(splt) == 3:
                    # tuple_word: (word, norm, pos)
                    tuple_word = (splt[0], splt[1], int(splt[2]))
                    docs[-1].words.append(tuple_word)
                    n -= 1

                elif      m > 0 and len(splt) == 3:
                    # sentence: (begin, end, sentence)
                    sentence = (int(splt[0]), int(splt[1]), splt[2]) 
                    docs[-1].sentences.append(sentence)
                    m -= 1

        if n or m: raise ValueError
        return docs

