#!/usr/bin/env python
# -*- coding: utf-8 -*-

from base64 import b64decode, b64encode
from zlib import compress, decompress
import numpy as np
import codecs
import sys
import re
import os

from index_builder import StrictIndexDocument, StrictIndex
from snippet_builder import SnippetBuilder


if __name__ == '__main__':
    # ------------------------------------------
    if   sys.argv[1] == '-index':
        
        if len(sys.argv) >= 4 and sys.argv[2] == '-urls':
            URLs = sys.argv[3]
        else:
            URLs = "./data/urls.txt"

        if len(sys.argv) >= 6 and sys.argv[4] == '-docs':
            DOCs = sys.argv[5]
        else:
            DOCs = "./data/docs-000.txt"
        # ------------------------------------------
        n = 30 # -1

        urls = []
        with codecs.open(URLs, 'r', encoding='utf-8') as f_urls:
            for line in f_urls.readlines()[:n]:
                if len(line.split()) > 1:
                    urls.append(line.split()[1])

        docs = []
        with codecs.open(DOCs, 'r', encoding='utf-8') as f_docs:
            for line in f_docs.readlines()[:n]:
                if len( line.split() ) > 1:
                    docs.append(decompress( b64decode(line.split()[1]) ))
                    # print docs[-1][:150].encode('cp866', 'ignore'), '\n\n'

        # ------------------------------------------
        index = StrictIndex(u'Lenta.ru20-StrictIndex.txt') # u'povarenok.ru30-StrictIndex.txt')
        indexed_docs = index.build(urls, docs, extacted_cache=True)
        index.dump(indexed_docs)
    # ------------------------------------------
    elif sys.argv[1] == '-snippet':

        if len(sys.argv) >= 4 and sys.argv[2] == '-queries':
            Queries = sys.argv[3]

            queries = []
            with codecs.open(Queries, 'r', encoding='utf-8') as f_queries:
                for line in f_queries.readlines():
                    splt = line.split('\t')
                    if len(splt) > 1:
                        queries.append( ( splt[0], int(splt[1]) ) )
        # ------------------------------------------
        else: # input from console

            print '\n\tinput=',
            if sys.platform.startswith('win'):
                query = unicode(sys.stdin.readline(), 'cp866')
            else:
                reload(sys)
                sys.setdefaultencoding('utf-8')
                query = unicode( sys.stdin.readline() )
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

            queries = []
            splt = query.split('\t')
            if splt > 1:
                queries += [( splt[0], int(splt[1]) )]
            else:
                print "Incorrect input!"

        # ------------------------------------------
        if queries:
            index = StrictIndex(u'Lenta.ru20-StrictIndex.txt') # u'povarenok.ru30-StrictIndex.txt') # 
            SB = SnippetBuilder(index)

            for query, doc_id in queries:

                if sys.platform.startswith('win'):
                    print (u"query= '%s'\n" % query).encode('cp866', 'ignore')
                    print SB.snippet(query, doc_id=doc_id).encode('cp866', 'ignore'), '\n\n'
                else:
                    print (u"query= '%s'\n" % query)
                    print SB.snippet(query, doc_id=doc_id), '\n\n'
        # ------------------------------------------
    else: print "Incorrect argument!"

