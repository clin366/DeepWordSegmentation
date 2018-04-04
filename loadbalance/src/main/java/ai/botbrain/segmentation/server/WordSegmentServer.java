package ai.botbrain.segmentation.server;

import ai.botbrain.segmentation.remotemodel.PosResult;
import ai.botbrain.segmentation.remotemodel.RemoteModel;
import ai.botbrain.segmentation.remotemodel.WordSegmentService;
import org.apache.commons.cli.*;
import org.apache.thrift.TException;
import org.apache.thrift.TProcessor;
import org.apache.thrift.protocol.TCompactProtocol;
import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TThreadPoolServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;
import org.apache.thrift.transport.TZlibTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Courage on 2018/3/27.
 */
public class WordSegmentServer implements WordSegmentService.Iface {
    private static final Logger logger = LoggerFactory.getLogger(WordSegmentServer.class);

    private List<RemoteModelWithWeight> remoteModelList;

    private WordSegmentServer(int size, String... args) {
        remoteModelList = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            remoteModelList.add(new RemoteModelWithWeight(args));
        }
    }

    @Override
    public boolean alive() throws TException {
        return true;
    }

    @Override
    public List<String> segmentText(String input) throws TException {
        RemoteModelWithWeight remoteModelWithWeight = getRemoteModel();
        try {
            return remoteModelWithWeight.segmentText(input);
        } finally {
            remoteModelWithWeight.decrementConnect();
        }
    }

    @Override
    public List<List<String>> segmentTexts(List<String> inputs) throws TException {
        RemoteModelWithWeight remoteModelWithWeight = getRemoteModel();
        try {
            return remoteModelWithWeight.segmentTexts(inputs);
        } finally {
            remoteModelWithWeight.decrementConnect();
        }
    }

    @Override
    public List<PosResult> posTagging(List<String> words) throws TException {
        RemoteModelWithWeight remoteModelWithWeight = getRemoteModel();
        try {
            return remoteModelWithWeight.posTagging(words);
        } finally {
            remoteModelWithWeight.decrementConnect();
        }
    }

    @Override
    public List<List<PosResult>> posTaggings(List<List<String>> wordsList) throws TException {
        RemoteModelWithWeight remoteModelWithWeight = getRemoteModel();
        try {
            return remoteModelWithWeight.posTaggings(wordsList);
        } finally {
            remoteModelWithWeight.decrementConnect();
        }
    }

    @Override
    public List<PosResult> segmentWithPosTagging(String input) throws TException {
        RemoteModelWithWeight remoteModelWithWeight = getRemoteModel();
        try {
            return remoteModelWithWeight.segmentWithPosTagging(input);
        } finally {
            remoteModelWithWeight.decrementConnect();
        }
    }

    @Override
    public List<List<PosResult>> segmentWithPosTaggings(List<String> inputs) throws TException {
        RemoteModelWithWeight remoteModelWithWeight = getRemoteModel();
        try {
            return remoteModelWithWeight.segmentWithPosTaggings(inputs);
        } finally {
            remoteModelWithWeight.decrementConnect();
        }
    }

    private static final class RemoteModelWithWeight extends RemoteModel {
        final AtomicInteger connect = new AtomicInteger(0);

        RemoteModelWithWeight(String... args) {
            super(args);
        }

        int getWeight() {
            return - connect.intValue();
        }

        RemoteModelWithWeight getAndIncrementConnect() {
            connect.incrementAndGet();
            return this;
        }

        void decrementConnect() {
            connect.decrementAndGet();
        }
    }

    private RemoteModelWithWeight getRemoteModel() {
        List<RemoteModelWithWeight> list = new ArrayList<>();
        int maxWeight = Integer.MIN_VALUE;
        for (RemoteModelWithWeight remoteModelWithWeight : remoteModelList) {
            int weight = remoteModelWithWeight.getWeight();
            if (weight > maxWeight) {
                maxWeight = weight;
                list.clear();
                list.add(remoteModelWithWeight);
            } else if (weight == maxWeight) {
                list.add(remoteModelWithWeight);
            }
        }
        return list.get(new Random().nextInt(list.size())).getAndIncrementConnect();
    }

    public static void main(String[] args) {
        Options options = new Options();
        options.addOption("h",false,"list help");
        options.addRequiredOption("p","port", true, "server port");
        options.addRequiredOption("s","size", true, "backend client size");
        options.addRequiredOption("c", "client", true, "backend client connect info");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd;
        try {
            cmd = parser.parse(options, args);
        } catch (MissingOptionException e) {
            logger.error(e.getMessage(), e);
            String str = "Botbrain";
            HelpFormatter hf = new HelpFormatter();
            hf.printHelp("WordSegmentServer", str, options, null);
            return;
        } catch (ParseException e) {
            logger.error(e.getMessage(), e);
            return;
        }

        if (cmd.hasOption("h")){
            String str = "Botbrain";
            HelpFormatter hf = new HelpFormatter();
            hf.printHelp("WordSegmentServer", str, options, null);
            return;
        }

        try {
            int port = Integer.parseInt(cmd.getOptionValue("p"));
            int size = Integer.parseInt(cmd.getOptionValue("s"));
            String[] clients = cmd.getOptionValues("c");

            logger.info("Thrift init start");
            WordSegmentServer handler = new WordSegmentServer(size, clients);
            TProcessor processor = new WordSegmentService.Processor<>(handler);
            TServerTransport tServerTransport = new TServerSocket(port);
            TServer server = new TThreadPoolServer(
                    new TThreadPoolServer.Args(tServerTransport)
                            .transportFactory(new TZlibTransport.Factory())
                            .protocolFactory(new TCompactProtocol.Factory())
                            .processor(processor)
                            .maxWorkerThreads(clients.length * 24)
            );
            logger.info("Thrift init finished");
            logger.info("Server start");
            server.serve();
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
        }
    }
}
