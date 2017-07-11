# -*- coding: utf-8 -*-
# @Author: Koth
# @Date:   2017-01-25 16:30:27
# @Last Modified by:   Koth
# @Last Modified time: 2017-01-31 11:52:33

# 此code用于构造 <实拍><美国><小飞机><迫降><繁忙><高速><吓坏><路上><司机> 类型的至LSTM POS model的训练集

import sys
import os
import w2v

totalLine = 0
longLine = 0

MAX_LEN = 50
totalChars = 0


class Sentence:
  def __init__(self):
    self.tokens = []
    self.markWrong = False

  def addToken(self, token):
    self.tokens.append(token)

  def generate_train_line(self, out, word_vob, char_vob):
    nl = len(self.tokens)
    if nl < 3:
      return
    wordi = []
    chari = []
    labeli = []
    if nl > MAX_LEN:
      nl = MAX_LEN
    for ti in range(nl):
      t = self.tokens[ti]
      idx = word_vob.GetWordIndex(t.token)
      wordi.append(str(idx))
      labeli.append(str(t.posTag))
      nc = len(t.chars)
      if nc > 5:
        lc = t.chars[nc - 1]
        t.chars[4] = lc
        nc = 5
      for i in range(nc):
        idx = char_vob.GetWordIndex(str(t.chars[i].encode("utf8")))
        chari.append(str(idx))
      for i in range(nc, 5):
        chari.append("0")
    for i in range(nl, MAX_LEN):
      wordi.append("0")
      labeli.append("0")
      for ii in range(5):
        chari.append(str(ii))
    line = " ".join(wordi)
    line += " "
    line += " ".join(chari)
    line += " "
    line += " ".join(labeli)
    out.write("%s\n" % (line))


class Token:
  def __init__(self, token, posTag):
    self.token = token
    ustr = unicode(token.decode('utf8'))
    self.chars = []
    for u in ustr:
      self.chars.append(u)
    self.posTag = posTag


def processToken(token, sentence, out, end, word_vob, char_vob, pos_vob):
  global totalLine
  global longLine
  global totalChars
  global MAX_LEN

  if token != "..":
    sentence.addToken(Token(token, 1))

  return True

def processLine(line, out, word_vob, char_vob, pos_vob):
  global totalLine
  line = line.strip()
  #line = line.decode('UTF-8')
  nn = len(line)
  start = 0
  sentence = Sentence()

  try:
    for i in range(nn):
      if line[i] == "<":
        for j in range(i + 1, nn):
          if line[j] == ">":
            token = line[i + 1 : j]
            if token != " ":
              processToken(token, sentence, out, False, word_vob, char_vob, pos_vob)
            if j == nn - 1:
              processToken("..", sentence, out, False, word_vob, char_vob, pos_vob)
            break
    sentence.generate_train_line(out, word_vob, char_vob)
    totalLine += 1
  except Exception as e:
    raise (e)
    pass

def loadPosVob(path, vob):
  fp = open(path, "r")
  for line in fp.readlines():
    line = line.strip()
    if not line:
      continue
    ss = line.split("\t")
    vob[ss[0]] = int(ss[1])
  pass

def main(argc, argv):
  global totalLine
  global longLine
  global totalChars
  if argc < 6:
    print("Usage:%s <word_vob> <char_vob> <pos_vob>  <dir> <output>" %
          (argv[0]))
    sys.exit(1)
  wvobPath = argv[1]
  cvobpath = argv[2]
  pvobPath = argv[3]
  rootDir = argv[4]
  word_vob = w2v.Word2vecVocab()
  word_vob.Load(wvobPath)
  char_vob = w2v.Word2vecVocab()
  char_vob.Load(cvobpath)
  posVob = {}
  loadPosVob(pvobPath, posVob)
  out = open(argv[5], "w")

  fp = open(rootDir, "r")

  for line in fp.readlines():
    line = line.strip()
    processLine(line, out, word_vob, char_vob, posVob)

  fp.close()
  out.close()
  
  print("total:%d, long lines:%d, chars:%d" %
        (totalLine, longLine, totalChars))

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
