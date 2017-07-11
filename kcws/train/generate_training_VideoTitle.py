# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-5-23 16:17:53
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-05-31 15:54:11

# 此code适用于处理生成<实拍><美国><小><飞机><迫降><繁忙><高速>< ><吓坏><路上><司机>类型的至LSTM的视频标题
# 运行跑 ./bazel-bin/kcws/train/generate_training_VideoTitle vec.txt <VideoTitle路径> all.txt

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

def processToken(token, sentence, out, vob):
  global totalLine
  global longLine
  global totalChars
  global MAX_LEN
  nn = len(token)

  if token != ".":
    sentence.addToken(token)
  else:
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
      if line[i] == "<":
        for j in range(i + 1, nn):
          if line[j] == ">":
            token = line[i + 1 : j]
            if token != " ":
              processToken(token, sentence, out, vob)
            if j == nn - 1:
              processToken(".", sentence, out, vob)
            break
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
  VideoFile = open(argv[2], "r")
  vob = w2v.Word2vecVocab()
  vob.Load(vobPath)
  out = open(argv[3], "a")
  count = 0

  for line in VideoFile.readlines():
      if count%100000 == 0:
          print ("Now is processing : " + str(count/100000) + " 十万")

      line = line.strip()
      processLine(line, out, vob)
      count += 1

  VideoFile.close()
  out.close()

  print("total:%d, long lines:%d, chars:%d" %
        (totalLine, longLine, totalChars))

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
