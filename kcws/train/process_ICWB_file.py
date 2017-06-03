#!/usr/bin/python  
# -*- coding: utf-8 -*-
# @Author : Yu Chen
# @Date: 2017-5-22 18:53:23
# @Last Modified by: Yu Chen
# @Last Modified time: 2017-5-31 11:03:33

# 此code用于处理生成videotitle文本的w2v生成，进行单字切分
# 运行 python kcws/train/process_VideoTitle_file.py <ICWB语料目录> pre_chars_for_w2v.txt

import sys
import re
import os

totalLine = 0
longLine = 0

def processLine(VideoTitle, out):

	global totalLine

	# 将每句话拆分为单字
	while True:
		char_seperate = ""
		line = VideoTitle.readline()
		totalLine += 1

		if not line:
			break

		if totalLine % 10000 == 0:
			print "Now is processing: " + str(totalLine/10000) + " 万"

		line = line.decode('utf8')

		for word in line:
			if word != " ":
				char_seperate += word + " "

		out.write("%s\n" %(char_seperate.encode('utf8')))

def main(argc, argv):

	global totalLine
	global longLine

	if argv < 3:
		print("Usage:%s <dir> <output>" % (argv[0]))
		sys.exit(1)

	ICWB_file = open(argv[1], "r")
	out = open(argv[2], "a")

	processLine(ICWB_file, out)

	ICWB_file.close()
	out.close()

	print("total:%d, long lines:%d" % (totalLine, longLine))

if __name__ == '__main__':
	main(len(sys.argv), sys.argv)
