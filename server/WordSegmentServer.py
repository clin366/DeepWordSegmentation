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
    def __init__(self, segment_model_path, pos_tag_model_path):
        vec_path = segment_model_path + "/vec.txt"
        segmentation_parameters_path = segment_model_path + "/parameters.txt"
        segmentation_crf_transition_matrix_path = segment_model_path + "/crf_transition_matrix.txt"
        segmentation_model_path = segment_model_path + '/SegmentModel'
        self.segment_method = segmentation(vec_path, segmentation_parameters_path, segmentation_crf_transition_matrix_path, segmentation_model_path)
        
        char_vec_path = pos_tag_model_path + "/char_vec.txt"
        word_vec_path = pos_tag_model_path + "/word_vec.txt"
        posTag_parameters_path = pos_tag_model_path + "/parameters.txt"
        posTag_crf_transition_matrix_path = pos_tag_model_path + '/crf_transition_matrix.txt'
        tag_path = pos_tag_model_path + "/tag.csv"
        posTag_model_path = pos_tag_model_path + '/PosTagModel'
        self.postag_method = posTag(char_vec_path, word_vec_path, posTag_parameters_path, posTag_crf_transition_matrix_path, tag_path, posTag_model_path)

    def alive(self):
        return True

    def segmentText(self, input):
        try:
            logger.info("segmentText:" + input)
            result = self.segment_method.generate_final_result(input)
            return result[0]
        except Exception as e:
            logger.error('%s' % e.message)

    def segmentTexts(self, inputs):
        # inputs is a list of string
        try:
            logger.info("segmentTexts error")
            result = self.segment_method.generate_final_result(inputs)
            return result
        except Exception as e:
            logger.error('%s' % e.message)

    def posTagging(self, words):
        try:
            logger.info("posTagging:" + str(words))
            postag_result = self.postag_method.posTagging_text(words)
            return postag_result[0]
        except Exception as e:
            logger.error('%s' % e.message)

    def posTaggings(self, wordsList):
        # wordsList is a list of list of string
        try:
            logger.info("posTaggings error")
            postag_result = self.postag_method.posTagging_text(wordsList)
            return postag_result
        except Exception as e:
            logger.error('%s' % e.message)

    def segmentWithPosTagging(self, input):
        try:
            logger.info("segmentWithPosTagging:" + input)
            segment_result = self.segment_method.generate_final_result(input)
            postag_result = self.postag_method.posTagging_text(segment_result)
            return postag_result[0]
        except Exception as e:
            logger.error('%s' % e.message)

    def segmentWithPosTaggings(self, inputs):
        # inputs is a list of string
        try:
            logger.info("segmentWithPosTagging error")
            segment_result = self.segment_method.generate_final_result(input)
            postag_result = self.postag_method.posTagging_text(segment_result)
            return postag_result
        except Exception as e:
            logger.error('%s' % e.message)

if __name__ == '__main__':
    handler = WordSegmentServiceHandler(segment_model_path=sys.argv[2], pos_tag_model_path=sys.argv[3])
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