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
    def __init__(self):
        """ """
        self.id = 0
        self.url = ''
        self.words = []      # (word,norm,position)
        self.sentenses = []  # (begin,end)
        self.n_sentenses = 0
        self.n_words = 0


class StrictIndex(object):
    """ index_builder 
            in   набор документов 
            out  прямой индекс
    """
    def __init__(self, fn_index=u'StrictIndex.txt', SS=SentenceSplitter()):
        """ """
        self.morph = pymorphy2.MorphAnalyzer()
        self.fn_index = fn_index
        self.re_extract_words = re.compile(ur'[^a-zа-яё0-9-]')
        self.re_repeat_spaces = re.compile(ur'[ ]+')
        self.re_set_paragraph = re.compile(ur'\r?\n')
        self.SS = SS

    # extractor='KeepEverythingExtractor'
    def build(self, urls=[], extractor='ArticleExtractor'):
        """ """
        docs = []
        doc_id = 0
        for URL in urls:
            doc = StrictIndexDocument()
            doc.id, doc.url = doc_id, URL
            doc_id += 1

            extractor = Extractor(extractor=extractor, url=URL)
            extracted_text = extractor.getText()

            words = self.re_extract_words.sub(u' ', extracted_text.lower())
            words = self.re_repeat_spaces.sub(u' ', words)

            for pos,word in enumerate(words):
                norm = self.morph.parse(word)[0].normal_form
                doc.words.append( (word, norm, pos) )

            paragraph = self.re_set_paragraph.sub(u' ', extracted_text)
            doc.sentenses = self.SS.find_sentenses(paragraph)

            docs.append(doc)
        return docs
    
    def dump(self, docs):
        """ """
        # sys.stdout = codecs.getwriter('cp866')(sys.stdout)
        with codecs.open(self.fn_index, 'w', encoding='utf-8') as f_index:
            for doc in docs:
                print >>f_index, u'%d\t%s\t%d\t%d\n' % (doc.id, doc.url, doc.n_words, doc.n_sentenses)

                for tuple in doc.words:
                    # tuple: (word,norm,position)
                    print >>f_index, u'%s\t%s\t%d' % (tuple[0], tuple[1], tuple[2])

                for sentence in doc.sentenses:
                    # sentence: (begin,end)
                    print >>f_index, u'%d\t%d' % (sentence[0], sentence[1])

                print >>f_index, '\n\n'

    def read(self):
        """ """
        docs = []
        n, m = 0, 0
        with codecs.open(self.fn_index, 'r', encoding='utf-8') as f_index:
            for line in f_index.readlines():
                line_split = line.strip().split() # (u'\t')

                if not (n or m) and line_split == 4:
                    doc = StrictIndexDocument()
                    (doc.id, doc.url, doc.n_words, doc.n_sentenses) = line_split
                    n = doc.n_words
                    m = doc.n_sentenses
                    docs.append(doc)

                elif      n > 0 and line_split == 3:
                    # tuple_word: (word, norm, pos)
                    tuple_word = ( line_split[0], line_split[1], int(line_split[2]) )
                    docs[-1].words.append(t_word)
                    n -= 1

                elif      m > 0 and line_split == 2:
                    # sentence: (begin, end)
                    sentence = ( int(line_split[0]), int(line_split[1]) ) 
                    docs[-1].sentenses.append(sentence)
                    m -= 1

        if n or m: raise ValueError
        return docs

