#encoding=UTF-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('gen-py')

from ai.botbrain.wordsegment import WordSegmentService

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

import time
import threading

import loggingUtils
import sendEmail

logger = loggingUtils.my_logging.get_my_log("./logs/WordSegmentServerAlarmer.log", 'WordSegmentServerAlarmer')

client = ""
t = ""

def OnTimer():
    logger.info("OnTimer")
    global client
    try:
        client.alive()
        logger.info("心跳正常")
    except Thrift.TException as tx:
        logger.info('%s' % tx.message)
        logger.info("分词服务掉线")
        sendEmail.send_mail("分词服务掉线", "不好啦，出大事啦，分词服务掉线啦!")
        raise tx
    global t
    t = threading.Timer(30.0, OnTimer)
    t.start()

def main():
    # Make socket
    transport = TSocket.TSocket(sys.argv[1], sys.argv[2])

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TFramedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    global client
    client = WordSegmentService.Client(protocol)

    # Connect!
    transport.open()

    global t
    t = OnTimer()

    while True:
        time.sleep(30)

    # Close!
    transport.close()

if __name__ == '__main__':
    try:
        main()
    except Thrift.TException as tx:
        logger.info('%s' % tx.message)
