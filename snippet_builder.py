#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pymorphy2
import codecs
import sys
import re
import os

from index_builder import StrictIndex, StrictIndexDocument


class SnippetBuilder(object):
    """ snippet_builder
            in   запрос и url/id документа 
            out  сниппет 
    """
    def __init__(self, index=StrictIndex()):
        """ """
        self.index = index

    def snippet(self, doc):
        """ """
        snippet = ''


        return snippet
        

