#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import codecs
import pickle
import sys
import re
import os

from utils import MyXML


class SentenceSplitter(object):
    """
        in   абзацы        через \n   utf-8
        out  предложения   через \n   utf-8
    """
    def __init__(self, separs = u'.?!', n_chars=5, fn_abbrs=[ u'data/abbr_rus.txt', u'data/abbr_eng.txt' ]):
        """ """
        self.tree = DecisionTreeClassifier()
        self.n_chars = n_chars
        self.separs = separs
        self.sections = []

        n = str(n_chars)
        self.re_sepr = re.compile(ur'(?=(.{'+ n + u'})?([' + separs + u'])(.{'+ n + u'})?)')
        # print self.re_sepr.pattern

        # self.re_sep1 = re.compile(ur'\.')
        # self.re_sep2 = re.compile(ur'\!')
        # self.re_sep3 = re.compile(ur'\?')
        # self.re_sep4 = re.compile(ur'\;')
        # self.re_sep5 = re.compile(ur'\:')
        # self.re_sep6 = re.compile(ur'\,')

        self.re_space = re.compile(ur'[ ]') # \t
        self.re_punkt = re.compile(ur'[.,!?:;]')
        self.re_brace = re.compile(ur'[()]')

        self.re_name = re.compile(ur'[A-ZА-ЯЁ][a-zа-яё]?\.[ ]?([A-ZА-ЯЁ][a-zа-яё]?\.[ ]?)[A-ZА-ЯЁ][a-zа-яё]+')
        self.re_date = re.compile(ur'\d{2,4}(?:\.|/)\d{2}(?:\.|/)\d{2,4}')
        self.re_site = re.compile(ur'(?:https?://(?:www\.)?)[a-z0-9-]+\.[a-z0-9-]+/.+')

        # сокращения
        self.re_abbrs = []
        for fn in fn_abbrs:
            with codecs.open(fn, 'r', encoding='utf-8') as f_abbr:
                re_abbrs = u'|'.join([ line.rstrip('\n') for line in f_abbr.readlines() ])
                # print re_abbrs[:160].encode('cp866', 'ignore'), '\n'
                self.re_abbrs.append( re.compile(re_abbrs) )
        return

    def features(self, vl_sections, separ_chars, vr_sections):
        """ """
        n = self.n_chars
        X = []
        # features
        for lf, sep, rt in zip (vl_sections, separ_chars, vr_sections):
            if lf or rt:
                feature = []
            
                # print ("'" + "'  '".join([lf, sep, rt]) + "'").encode('cp866', 'ignore')

                # v  число слов в контекстах
                # v  число букв в последних 2х словах
                # v  точка в где-то в контекстах
                # -----------------------------------------------
                if lf:
                    for c in lf:
                        feature.append( 1 if c.isupper() else 0 )
                        feature.append( 1 if c.islower() else 0 )
                        feature.append( 1 if c.isdigit() else 0 )
                        feature.append( 1 if self.re_punkt.match(c) is not None else 0 )
                        feature.append( 1 if self.re_space.match(c) is not None else 0 )
                        feature.append( 1 if self.re_brace.match(c) is not None else 0 )
                    if len(lf) < n: # ?????
                        feature += [0,0,0,0,0,0] * (n - len(lf))
                else:
                    feature += [0,0,0,0,0,0] * n
                
                # -----------------------------------------------
                if rt:
                    for c in rt:
                        feature.append( 1 if c.isupper() else 0 )
                        feature.append( 1 if c.islower() else 0 )
                        feature.append( 1 if c.isdigit() else 0 )
                        feature.append( 1 if self.re_punkt.match(c) is not None else 0 )
                        feature.append( 1 if self.re_space.match(c) is not None else 0 )
                        feature.append( 1 if self.re_brace.match(c) is not None else 0 )
                    if len(rt) < n: # ?????
                        feature += [0,0,0,0,0,0] * (n - len(rt))
                else:
                    feature += [0,0,0,0,0,0] * n
                                    
                # -----------------------------------------------
                for s in self.separs:
                    feature.append(  1 if sep == s else 0 )
                feature.append( 1 if sep.isupper() else 0 )
                feature.append( 1 if sep.islower() else 0 )
                feature.append( 1 if self.re_punkt.match(sep) is not None else 0 )

                # -----------------------------------------------
                peace = (lf if lf else u'') + sep + (rt if rt else u'')

                nm = self.re_name.match(peace) is not None
                # if nm: print (peace).encode('cp866', 'ignore') # !!!
                feature.append( 1 if nm else 0 )

                dt = self.re_date.match(peace) is not None
                # if dt: print (peace).encode('cp866', 'ignore') # !!!
                feature.append( 1 if dt else 0 )

                st = self.re_site.match(peace) is not None
                # if st: print (peace).encode('cp866', 'ignore') # !!!
                feature.append( 1 if st else 0 )
                
                # -----------------------------------------------
                if sep == '.':
                    peace = (lf if lf else u'') + sep
                    abbrs = [ (1 if re.match(peace) else 0) for re in self.re_abbrs ]
                    feature.append( 1 if any(abbrs) else 0 )
                else:
                    feature.append(0)

                # if len(feature) != 70: print len(feature),   # !!!
                # -----------------------------------------------
                sample = feature
                X.append(sample)
        return  np.mat(X)

    def fit(self, corpus=None, corpus_pos_s=None, fn_fit_corpus=u'data/sentences.xml'):
        """ """
        fn = re.split('\\/', fn_fit_corpus)[-1]
        fn = re.split(r'\.', fn)[-2]

        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.isfile('data/' + fn + '-tree.pkl'):
            vl_sections = []
            separ_chars = []
            vr_sections = []

            if corpus is None or corpus_pos_s is None:
                myxml = MyXML()
                if not os.path.exists('data/' + fn + '-cached.txt'):
                    corpus, corpus_pos_s = myxml.read_xml(fn_fit_corpus)
                    myxml.text_dump('data/' + fn + '-cached.txt', corpus, corpus_pos_s)
                else:
                    corpus, corpus_pos_s = myxml.text_read('data/' + fn + '-cached.txt')

            n = self.n_chars
            for p in corpus_pos_s:
                vl_sections.append( corpus[p-n:p] ) 
                separ_chars.append( corpus[p] ) 
                vr_sections.append( corpus[p+1:p+n+1] )

            sl_sects, sr_sects = set(vl_sections), set(vr_sections)
            for m in self.re_sepr.finditer(corpus):
                if      (m.group(1)  or  m.group(3)) \
                    and (m.group(1) not in sl_sects) \
                    and (m.group(3) not in sr_sects):

                    vl_sections.append( m.group(1) )
                    separ_chars.append( m.group(2) )
                    vr_sections.append( m.group(3) )

            X = self.features(vl_sections, separ_chars, vr_sections)
            y = np.mat( [1] * len(corpus_pos_s) + [0] * (X.shape[0] - len(corpus_pos_s)) )
            self.tree.fit(X, y.T)

            str_tree = pickle.dumps(self.tree)
            with open('data/' + fn + '-tree.pkl', 'w') as f_tree:
                f_tree.write(str_tree)
            # print len(str_tree)

        else:
            with open('data/' + fn + '-tree.pkl', 'r') as f_tree:
                str_tree = f_tree.read()
                self.tree = pickle.loads(str_tree)
        return

    def predict(self, text):
        """ """
        vl_sections = []
        separ_chars = []
        vr_sections = []

        positions = []
        for m in self.re_sepr.finditer(text):
            if m.group(1) or m.group(3):
                positions.append( m.start() \
                                  + (len(m.group(1)) if m.group(1) else 0) \
                                  +  len(m.group(2)) )
                # str1 =  u'' + (m.group(1) if m.group(1) else '') \
                #             +  m.group(2) \
                #             + (m.group(3) if m.group(3) else '')
                # print ( str1 ).encode('cp866', 'ignore')
                vl_sections.append( m.group(1) if m.group(1) else u'')
                separ_chars.append( m.group(2) )
                vr_sections.append( m.group(3) if m.group(3) else u'')

        T  = self.features(vl_sections, separ_chars, vr_sections)
        py = self.tree.predict(T)

        py = np.array(py,dtype=bool)
        positions = np.array(positions)[py]

        sentences = []
        for splt in np.split(list(text), list(positions)):
            sentences.append( ''.join(splt) )
        return sentences, positions



def main():
    test_str  = u' Конечно, в рамках газетной статьи невозможно сделать обзор сотни докладов, поэтому мы рекомендуем посетить Интернет-страницу конференции http://www.Ict.nsc.ru/ws/mol2000/, на которой размещены программа мероприятия и тезисы докладов.'
    test_str += u' В 11.45 дали слово Кудрину , но он всё не шёл.'
    test_str += u' выполнение 12 тепловозам усиленного ТР-1 с применением средств диагностики вместо ТР-2 дало экономический эффект 221,8 тыс. руб.'
    test_str += u' Только вот ради чего ?!'
    test_str += u' Сегодня снял 5 тыс. руб.'
    test_str += u' В связи с этим первый интервал пробегов был принят равным 350...700 тыс. км (середина интервала - 525 тыс. км), второй интервал -- 700...1050 тыс. км (середина интервала - 875 тыс. км) и третий интервал 1050...1400 тыс. км (середина интервала -- 1225 тыс. км).'
    test_str += u' Не в лесе и не в медицине дело...'
    test_str += u' Дело в том, что по всяким планам «пятилеток» и заданиям ЦК советский военный комплекс создавал ядерное оружие с запасом на пять и более(!) ядерных войн.'
    test_str += u' Очень классные предложения =))'
    test_str += u' Конец по всяким планам конца.'

    ss = SentenceSplitter()
    ss.fit()


    sentences, positions = ss.predict(test_str)

    # sentences = splitter.split_sentences(test_str)
    for sent in sentences:
        print '#', sent.encode('cp866', 'ignore')


if __name__ == '__main__':
    main()
