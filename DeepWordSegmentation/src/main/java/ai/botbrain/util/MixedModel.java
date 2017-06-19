package ai.botbrain.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Courage on 2017/6/19.
 */
public class MixedModel {
    private static final Logger logger = LoggerFactory.getLogger(MixedModel.class);

    private Process process = null;
    private BufferedReader bufferedReader = null;
    private BufferedWriter bufferedWriter = null;
    private boolean inited;

    public MixedModel() {
        inited = false;
    }

    public boolean init() {
        if (inited) {
            return false;
        }
        inited = true;
        try {
            process = new ProcessBuilder().command("python", "/Users/vongola/Desktop/DeepLearning/filter/filter.py").redirectErrorStream(true).start();
            bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            bufferedWriter = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            return true;
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
            return false;
        }
    }

    public List<String> segment(String input) {
        List<String> result = new ArrayList<String>();
        try {
            bufferedWriter.write(input);
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                if (line.contains("Some flag mean one end")) {
                    break;
                }
                result.add(line);// handle the output
            }
            return result;
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
        return result;
    }

    public void destory() {
        try {
            if (process != null) {
                // input or do something to end the process
            }
            if (bufferedReader != null) {
                bufferedReader.close();
                bufferedReader = null;
            }
            if (bufferedWriter != null) {
                bufferedWriter.close();
                bufferedWriter = null;
            }
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
    }
}
