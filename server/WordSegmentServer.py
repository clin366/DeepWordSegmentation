#encoding=UTF-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('gen-py')

from ai.botbrain.wordsegment import WordSegmentService
from ai.botbrain.wordsegment.ttypes import PosResult

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import loggingUtils

logger = loggingUtils.my_logging.get_my_log("./logs/WordSegmentServer.log", 'WordSegmentServer')

class WordSegmentServiceHandler:
    def __init__(self):
        pass

    def alive(self):
        return True

    def segmentText(self, input):
        try:
            logger.info("segmentText:" + input)
            result = []
            result.append("Segment")
            result.append("Text")
            result.append(input)
            return result
        except Exception as e:
            logger.error('%s' % e.message)

    def posTagging(self, words):
        try:
            logger.info("segmentWithPosTagging:" + str(words))
            result = []
            count = 1
            for word in words:
                posResult = PosResult(word, "Tag" + str(count))
                count += 1
                result.append(posResult)
            return result
        except Exception as e:
            logger.error('%s' % e.message)

    def segmentWithPosTagging(self, input):
        try:
            logger.info("segmentWithPosTagging:" + input)
            result = []
            posResult = PosResult("segmentWithPosTagging", "Tag1")
            result.append(posResult)
            posResult = PosResult(input, "Tag2")
            result.append(posResult)
            return result
        except Exception as e:
            logger.error('%s' % e.message)

if __name__ == '__main__':
    handler = WordSegmentServiceHandler()
    processor = WordSegmentService.Processor(handler)
    transport = TSocket.TServerSocket(port=sys.argv[1])
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(
    #     processor, transport, tfactory, pfactory)
    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    server.setNumThreads(20)

    logger.info('Starting the server...')
    try:
        server.serve()
    except Exception as e:
        logger.error('%s' % e.message)
    logger.info('done.')
