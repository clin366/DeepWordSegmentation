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

from segmentation import *
from posTag import *

logger = loggingUtils.my_logging.get_my_log("./logs/WordSegmentServer.log", 'WordSegmentServer')

class WordSegmentServiceHandler:
    def __init__(self):
        pass

    def alive(self):
        return True

    def segmentText(self, input):
        try:
            logger.info("segmentText:" + input)
            segment_method = segmentation()
            result = segment_method.generate_char_result(input)
            return result
        except Exception as e:
            logger.error('%s' % e.message)

    def posTagging(self, words):
        try:
            logger.info("segmentWithPosTagging:" + str(words))
            result = []
            postag_method = posTag()
            postag_result = postag_method.posTagging_text(words)
            count = 0
            for word in words:
                posResult = PosResult(word, postag_result[count])
                result.append(posResult)
                count += 1
            return result
        except Exception as e:
            logger.error('%s' % e.message)

    def segmentWithPosTagging(self, input):
        try:
            logger.info("segmentWithPosTagging:" + input)
            result = []
            segment_method = segmentation()
            postag_method = posTag()
            segment_result = segment_method.generate_char_result(input)
            postag_result = postag_method.posTagging_text(segment_result)
            count = 0
            for word in segment_result:
                posResult = PosResult(word, postag_result[count])
                result.append(posResult)
                count += 1
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
