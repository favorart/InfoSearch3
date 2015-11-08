#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import codecs
import sys
import re
import os


def lst_unique(lst):
    seen = set()
    return [e for e in lst if e not in seen and not seen.add(e)]


class SentenceSplitter(object):
    """
        in   абзацы        через \n   utf-8
        out  предложения   через \n   utf-8
    """
    def __init__(self, separs = u'.?!', n_chars=5, fn_abbrs=[ u'data/abbr_rus.txt', u'data/abbr_eng.txt' ]):
        """ """
        self.n_chars = n_chars
        self.separs = separs
        self.re_sepr = re.compile(ur'(?=(.{0,'+ str(self.n_chars) + u'})([' + separs + u'])(.{0,'+ str(self.n_chars) + u'}))')
        self.sections = []

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

        # # сокращения
        # self.re_abbrs = u''
        # for fn in fn_abbrs:
        #     with codecs.open(fn, 'r', encoding='utf-8') as f_abbr:
        #         for line in f_abbr.readlines():
        #             print u'.*' + line.strip()[-1].encode('cp866', 'ignore')
        #             self.abbrs.append( re.compile('.*' + line.strip()[-1])  )

        self.tree = DecisionTreeClassifier()
        
    def make_features(self, vl_sections, separ_chars, vr_sections):
        """ """
        n = self.n_chars
        X = []
        # features
        for lf, sep, rt in zip (vl_sections, separ_chars, vr_sections):
            feature = []
            
            # число слов в контекстах
            # число букв в последних 2х словах
            # точка в где-то в контекстах
            # сокращения

            feature.append( len(lf) )
            for i in xrange(n - len(lf)):
                feature += [0,0,0,0,0,0]
            for c in lf:
                feature.append( 1 if c.isupper() else 0 )
                feature.append( 1 if c.islower() else 0 )
                feature.append( 1 if c.isdigit() else 0 )
                feature.append( 1 if self.re_punkt.match(c) is not None else 0 )
                feature.append( 1 if self.re_space.match(c) is not None else 0 )
                feature.append( 1 if self.re_brace.match(c) is not None else 0 )

            # abbrs = []
            # for abbr in self.re_abbrs:
            #     abbrs.append( 1 if abbr.match.match(c) is not None else 0 )
            # feature.append( 1 if any(abbrs) else 0 )

            feature.append( len(rt) )
            for c in rt:
                feature.append( 1 if c.isupper() else 0 )
                feature.append( 1 if c.islower() else 0 )
                feature.append( 1 if c.isdigit() else 0 )
                feature.append( 1 if self.re_punkt.match(c) is not None else 0 )
                feature.append( 1 if self.re_space.match(c) is not None else 0 )
                feature.append( 1 if self.re_brace.match(c) is not None else 0 )
            for i in xrange(n - len(rt)):
                feature += [0,0,0,0,0,0]
              
            for s in self.separs:
                feature.append(  1 if sep == s else 0 )
            feature.append( 1 if sep.isupper() else 0 )
            feature.append( 1 if sep.islower() else 0 )
            feature.append( 1 if self.re_punkt.match(sep) is not None else 0 )

            nm = self.re_name.match( lf + sep + rt ) is not None
            feature.append( 1 if nm else 0 )

            sample = feature
            X.append(sample)

        X = np.mat(X)
        return X
        
    def fit(self, text, poss):
        """ """
        vl_sections = []
        separ_chars = []
        vr_sections = []

        n = self.n_chars
        for p in poss:
            vl_sections.append( text[p-n:p] ) 
            separ_chars.append( text[p] ) 
            vr_sections.append( text[p+1:p+n+1] )

        sl_sects, sr_sects = set(vl_sections), set(vr_sections)
        matches = self.re_sepr.findall (text)
        for sl in matches:
            if( sl[0] not in sl_sects and sl[2] not in sr_sects ):
                vl_sections.append( sl[0] )
                separ_chars.append( sl[1] )
                vr_sections.append( sl[2] )

        y = np.mat( [1] * len(poss) + [0] * (len(separ_chars) - len(poss)) )
        X = self.make_features(vl_sections, separ_chars, vr_sections)
        self.tree.fit(X,y.T)
        return

    def predict(self, text):
        """ """
        vl_sections = []
        separ_chars = []
        vr_sections = []

        # matches = self.re_sepr.findall(text)
        # for m in p.finditer('a1b2c3d4'):
        #     print m.start(), m.group()

        positions = []
        for m in self.re_sepr.finditer(text):
            positions.append( m.start() + len(m.group(1)) + len(m.group(2)) )
            # print (m.group(1)  +  m.group(2) + m.group(3)).encode('cp866', 'ignore')
            vl_sections.append( m.group(1) )
            separ_chars.append( m.group(2) )
            vr_sections.append( m.group(3) )

        # sentenses = re.split(ur'[' + self.separs + u']', text)
        # for i in xrange(len(sentenses)):
        #     sentenses[i] += separ_chars[i]
        #     # print sentenses[i][-20:].encode('cp866')

        T  = self.make_features(vl_sections, separ_chars, vr_sections)
        py = self.tree.predict(T)

        # for i,p in enumerate(py):
        #     if not p:
        #         sentenses[i] += sentenses[i+1]
        #         del sentenses[i+1]

        py = np.array(py,dtype=bool)
        positions = np.array(positions)[py]
        
        sentences = []
        for splt in np.split(list(text), list(positions)):
            sentences.append( ''.join(splt) )
        return sentences, positions
