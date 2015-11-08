#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs
import sys
import re
import os

# from index_builder import StrictIndexDocument, StrictIndex
# from sentence_splitter import SentenceSplitter
# from snippet_builder import SnippetBuilder
# from utils import MyXML


URL = u'https\://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_\
%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1\
%81%D0%BF%D1%80%D0%BE%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%\
D1%8F_%D0%BE%D1%88%D0%B8%D0%B1%D0%BA%D0%B8'
URL = u'http\://www.povarenok.ru/articles/show/8243/'


if __name__ == '__main__':

    with codecs.open(u'data/сокр2.txt', 'r', encoding='utf-8') as f:
        with codecs.open(u'data/abbr_rus.txt', 'w', encoding='utf-8') as f2:
            lines = list(set( f.readlines() ))
            lines.sort()
            f2.write ( u''.join( lines ) )

#if 1:
#    myxml = MyXML()
#    # text, poss = myxml.read_xml(u'data/sentences.xml')
#    # myxml.text_dump(u'data/sentences.txt', text, poss)

#    text, poss = myxml.text_read(u'data/sentences.txt')

#    # with codecs.open(u'data/outs.txt', 'w',encoding='utf-8') as f_out:
#    #     for p in poss:
#    #         print >>f_out, p, text[p]

#    text1, poss1 = myxml.text_read(u'data/sentences1.txt')
#    SS = SentenceSplitter(n_chars=5)
#    SS.fit(text, poss)
#    sentences, poss2 = SS.predict(text1)#    with codecs.open(u'data/outs.txt', 'w',encoding='utf-8') as f_out:
#        print >>f_out, u','.join( [ str(p) for p in poss1 ] )
#        print >>f_out, u','.join( [ str(p) for p in poss2 ] )
#        for s in sentences:
#            print >>f_out, s, '\n'#    index = StrictIndex()#    with codecs.open(sys.argv[1], 'r', encoding='utf-8') as f_urls:#        urls = f_urls.readlines()#    docs = index.build(urls)#    index.dump(docs)
