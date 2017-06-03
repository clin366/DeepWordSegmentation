# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-5-23 16:17:53
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-05-31 16:54:11

# 此code适用于处理生成ICWB语料的训练集
# 运行跑

import sys
import os
import w2v

totalLine = 0
longLine = 0

MAX_LEN = 80
totalChars = 0

class Setence:
  def __init__(self):
    self.tokens = []
    self.chars = 0

  def addToken(self, t):
    self.chars += len(t)
    self.tokens.append(t)

  def clear(self):
    self.tokens = []
    self.chars = 0

  # label -1, unknown
  # 0-> 'S'
  # 1-> 'B'
  # 2-> 'M'
  # 3-> 'E'
  def generate_tr_line(self, x, y, vob):
    for t in self.tokens:
      if len(t) == 1:
        x.append(vob.GetWordIndex(str(t[0].encode("utf8"))))
        y.append(0)
      else:
        nn = len(t)
        for i in range(nn):
          x.append(vob.GetWordIndex(str(t[i].encode("utf8"))))
          if i == 0:
            y.append(1)
          elif i == (nn - 1):
            y.append(3)
          else:
            y.append(2)

def processToken(token, sentence, out, end, vob):
  global totalLine
  global longLine
  global totalChars
  global MAX_LEN
  nn = len(token)

  if token != "。":
    sentence.addToken(token)

  if token == "。" or end:
    if sentence.chars > MAX_LEN:
      longLine += 1
    else:
      x = []
      y = []
      totalChars += sentence.chars
      sentence.generate_tr_line(x, y, vob)
      nn = len(x)
      assert (nn == len(y))

      # 不足80的后面全补上0
      for j in range(nn, MAX_LEN):
        x.append(0)
        y.append(0)

      line = ''
      for i in range(MAX_LEN):
        if i > 0:
          line += " "
        line += str(x[i])
      for j in range(MAX_LEN):
        line += " " + str(y[j])
      out.write("%s\n" % (line))
    totalLine += 1
    sentence.clear()

def processLine(line, out, vob):
  line = line.strip()
  line = line.decode('UTF-8')
  nn = len(line)
  start = 0
  sentence = Setence()

  try:

    for i in range(nn):
      if line[i] == " ":
        token = line[start:i]
        processToken(token, sentence, out, False, vob)
        start = i + 1

    if start < nn:
      token = line[start:]
      processToken(token, sentence, out, True, vob)

  except Exception as e:
    pass

def main(argc, argv):
  global totalLine
  global longLine
  global totalChars

  if argc < 4:
    print("Usage:%s <vob> <dir> <output>" % (argv[0]))
    sys.exit(1)

  vobPath = argv[1]
  ICWBFile = open(argv[2], "r")
  vob = w2v.Word2vecVocab()
  vob.Load(vobPath)
  out = open(argv[3], "a")
  count = 0

  for line in ICWBFile.readlines():
      if count % 10000 == 0:
        print("Now is processing " + str(count/10000) + " 万")
      
      count += 1
      line = line.strip()
      processLine(line, out, vob)

  ICWBFile.close()
  out.close()

  print("total:%d, long lines:%d, chars:%d" %
        (totalLine, longLine, totalChars))

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
