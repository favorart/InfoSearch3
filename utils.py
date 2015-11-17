#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs
import sys
import re
import os

import xml.etree.ElementTree


class MyXML(object):
    """ """
    def read_xml(self, fn_xml=u'sentences.xml'):
        """ """
        xml_root = xml.etree.ElementTree.parse(fn_xml).getroot()

        text = u''
        poss = []
        size = 0
        for txt in xml_root:
            for paragraph in txt[1]:
                for sentence in paragraph:
                    size += 1 + len(sentence[0].text)
                    text += u' ' +  sentence[0].text
                    poss.append( size - 1 )
                # text = text[:-1] + u'\n'
                text += u'\n'
                size += 1
            text += u'\n'
            size += 1

        return (text, poss)

    def text_dump(self, fn_out, text, poss):
        """ """
        with codecs.open(fn_out, 'w', encoding='utf-8') as f_out:
            print >>f_out, ','.join([ str(p) for p in poss ])
            print >>f_out, text

    def text_read(self, fn_in):
        """ """
        with codecs.open(fn_in, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            poss = [ int(p) for p in lines[0].split(u',') ]
            text = u''.join(  lines[1:] )
        return (text, poss)

