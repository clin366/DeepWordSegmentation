#encoding=UTF-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('gen-py')

from ai.botbrain.wordsegment import WordSegmentService

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.transport import TZlibTransport
from thrift.protocol import TCompactProtocol

import time
import threading
import traceback

import loggingUtils
import sendEmail

logger = loggingUtils.my_logging.get_my_log("./logs/WordSegmentServerAlarmer.log", 'WordSegmentServerAlarmer')

thriftClients = []
lostClients = []
t = ""

class ThriftClient():
    def __init__(self, ip, port):
        logger.info('ip=' + str(ip) + ' port=' + str(port))
        self.ip = ip
        self.port = port
        self.initConnect()

    def initConnect(self):
        socketTransport = TSocket.TSocket(self.ip, self.port)
        self.transport = TZlibTransport.TZlibTransport(socketTransport)
        self.transport.open()
        protocol = TCompactProtocol.TCompactProtocol(self.transport)
        self.client = WordSegmentService.Client(protocol)

    def close(self):
        self.transport.close()

def closeClients():
    global thriftClients
    for thriftClient in thriftClients:
        thriftClient.close()

def OnTimer():
    logger.info("OnTimer")
    global lostClients
    fixedClients = []
    for lostClient in lostClients:
        try:
            lostClient.initConnect()
            lostClient.client.alive()
            logger.info(str(lostClient.ip) + " " + str(lostClient.port) + " 服务恢复")
            sendEmail.send_mail(str(lostClient.ip) + " " + str(lostClient.port) + "分词服务上线", "哟哟，分词服务恢复了哟！")
            fixedClients.append(lostClient)
        except Thrift.TException as tx:
            logger.error('%s' % tx.message)
            logger.error(str(lostClient.ip) + " " + str(lostClient.port) + " 服务未恢复")

    global thriftClients
    for fixedClient in fixedClients:
        thriftClients.append(fixedClient)
        lostClients.remove(fixedClient)

    errorClients = []
    for thriftClient in thriftClients:
        try:
            thriftClient.client.alive()
            logger.info("心跳正常")
        except Thrift.TException as tx:
            logger.error("分词服务掉线")
            logger.error('%s' % tx.message)
            logger.error(traceback.format_exc())
            sendEmail.send_mail(str(thriftClient.ip) + " " + str(thriftClient.port) + "分词服务掉线", "不好啦，出大事啦，分词服务掉线啦!")
            errorClients.append(thriftClient)

    for errorClient in errorClients:
        thriftClients.remove(errorClient)
        lostClients.append(errorClient)

    global t
    t = threading.Timer(30.0, OnTimer)
    t.start()

def main():
    i = 1
    global thriftClients
    logger.info("开始初始化client")
    while i < len(sys.argv):
        thriftClients.append(ThriftClient(sys.argv[i], sys.argv[i + 1]))
        i += 2
    logger.info("初始化所有client成功")

    global t
    t = OnTimer()

    while True:
        time.sleep(30)

    closeClients()

if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        logger.info('%s' % tx.message)
