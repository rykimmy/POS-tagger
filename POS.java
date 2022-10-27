import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * This file holds all the methods pertaining to POS tagging
 * including training tables, viterbi, and viterbi testing methods
 * @author Ryan Kim (TA: Caroline Hall)
 * @author Anand McCoole (TA: Jason Pak)
 */

public class POS {
    Map<String, Map<String, Double>> tTable; // Transition Table: maps state (pos) to map of next pos and their scores
    Map<String, Map<String, Double>> oTable; // Observation Table: maps pos to map of words to their scores
    Map<String, Integer> tTotals; // Transition Totals Table
    Map<String, Integer> oTotals; // Observation Totals Table

    public POS() {
        tTable = new HashMap<>();
        oTable = new HashMap<>();
        tTotals = new HashMap<>();
        oTotals = new HashMap<>();
    }

    /**
     * Reads files (sentences + tags) to create our observation and transition tables for the purpose of training
     * Also calculates the scores of each word/POS in log form
     * @throws IOException
     */
    public void trainTables() throws IOException {
        BufferedReader sentenceReader = new BufferedReader(new FileReader("PS5/brown-test-sentences.txt"));
        BufferedReader tagReader = new BufferedReader(new FileReader("PS5/brown-test-tags.txt"));

        // Declares variables to hold the sentence as a String and as a list of Strings (each word)
        String sentenceLine;
        String[] sentenceSpline;

        // Declares variables to hold the line of tags as a String and as a list of Strings (each tag)
        String tagLine;
        String[] tagSpline;

        // Continues running until everything in file has been read
        while ((sentenceLine = sentenceReader.readLine()) != null) {

            // Switches to lower case
            sentenceLine = sentenceLine.toLowerCase();
            tagLine = tagReader.readLine().toLowerCase();

            // Splits lines into individual words
            sentenceSpline = sentenceLine.split(" ");
            tagSpline = tagLine.split(" ");

            // For each word in a line
            for (int i = 0; i < tagSpline.length; i++) {
                // If the Map oTable does not have POS tag
                if (!oTable.containsKey(tagSpline[i])) {
                    oTable.put(tagSpline[i], new HashMap<>());
                    oTable.get(tagSpline[i]).put(sentenceSpline[i], 1.0);
                }
                // If the Map oTable already has the POS tag
                else {
                    if (oTable.get(tagSpline[i]).containsKey(sentenceSpline[i])) {
                        double currFreq = oTable.get(tagSpline[i]).get(sentenceSpline[i]);
                        oTable.get(tagSpline[i]).put(sentenceSpline[i], currFreq + 1.0);
                    }
                    if (!oTable.get(tagSpline[i]).containsKey(sentenceSpline[i])) {
                        oTable.get(tagSpline[i]).put(sentenceSpline[i], 1.0);
                    }
                }

                // If first POS in the line, sets prev to "#"
                String prev;
                if (i == 0) {
                    prev = "#";
                } else prev = tagSpline[i - 1];

                // If the Map tTable doesn't have POS tag
                if (!tTable.containsKey(prev)) {
                    tTable.put(prev, new HashMap<>());
                    tTable.get(prev).put(tagSpline[i], 1.0);
                }

                // If the Map tTable already has the POS tag
                else {
                    if (tTable.get(prev).containsKey(tagSpline[i])) {
                        double currFreq = tTable.get(prev).get(tagSpline[i]);
                        tTable.get(prev).put(tagSpline[i], currFreq + 1.0);
                    }
                    if (!tTable.get(prev).containsKey(tagSpline[i])) {
                        tTable.get(prev).put(tagSpline[i], 1.0);
                    }
                }
            }

            // Create transition totals table
            for (String s1 : tTable.keySet()) {
                int sum = 0;
                for (String s2 : tTable.get(s1).keySet()) {
                    sum += tTable.get(s1).get(s2);
                }
                tTotals.put(s1, sum);
            }

            // Create observation totals table
            for (String s1 : oTable.keySet()) {
                int sum = 0;
                for (String s2 : oTable.get(s1).keySet()) {
                    sum += oTable.get(s1).get(s2);
                }
                oTotals.put(s1, sum);
            }
        }
        // Re-update values in oTable to the logarithm of each probability
        for (String pos : oTable.keySet()) {
            for (String word : oTable.get(pos).keySet()) {
                double score = Math.log(oTable.get(pos).get(word) / oTotals.get(pos));
                oTable.get(pos).put(word, score);
            }
        }
        // Re-update values in tTable to the logarithm of each probability
        for (String pos : tTable.keySet()) {
            for (String nextPos : tTable.get(pos).keySet()) {
                double score = Math.log(tTable.get(pos).get(nextPos) / tTotals.get(pos));
                tTable.get(pos).put(nextPos, score);
            }
        }
        // Close buffered readers
        sentenceReader.close();
        tagReader.close();
    }

    /**
     * Uses the training tables to run viterbi algorithm and tag words
     * @param sentence, the specific line being read
     * @return rline, returns the sentence with the corresponding tags generated by our viterbi algorithm
     * @throws IOException
     */
    public String viterbi(String sentence) throws IOException {
        String line;
        String[] spline;

        Set<String> currentStates;
        Map<String, Double> currentScores;
        Map<String, String> nextStates;
        Map<String, Double> nextScores;

        line = sentence.toLowerCase();
        spline = line.split(" ");

        // Initializes currentStates and currentScores | also hard-codes the start for every sentence
        currentStates = new HashSet<>();
        currentScores = new HashMap<>();
        currentStates.add("#");
        currentScores.put("#", 0.0);

        // Initialize our back tracing structure
        ArrayList<Map<String, String>> backTrace = new ArrayList<>();

        // Iterates through every word in the sentence and for each word, initializes new nextState + nextScores
        for (int i = 0; i < spline.length; i++) {
            nextStates = new HashMap<>();
            nextScores = new HashMap<>();

            for (String pos : currentStates) {
                for (String nextPOS : tTable.get(pos).keySet()) {
                    nextStates.put(nextPOS, pos);

                    // If word is not in nextPOS key set, we give it an unseen score
                    double unseenScore;
                    unseenScore = oTable.get(nextPOS).getOrDefault(spline[i], -100.0);

                    double nextScore = currentScores.get(pos) + tTable.get(pos).get(nextPOS) + unseenScore;

                    // Updates if nextScores doesn't contain nextPOS, or if nextScore > nextScores[nextPOS]
                    if (!nextScores.containsKey(nextPOS) || nextScore > nextScores.get(nextPOS)) {
                        nextScores.put(nextPOS, nextScore);

                        // Adds most likely transition state to backTrace
                        if (backTrace.size() == i) {
                            backTrace.add(i, new HashMap<>());
                            backTrace.get(i).put(nextPOS, pos);
                        } else {
                            backTrace.get(i).put(nextPOS, pos);
                        }
                    }
                }
            }

            // Updates currentStates and currentScores after every word
            currentStates = new HashSet<>();
            currentStates.addAll(nextStates.keySet());
            currentScores = nextScores;
        }

        // When last observation is made, we get the POS with the best viterbi score
        String bestScore = "";
        for (String p : currentScores.keySet()) {
            if (bestScore.equals("") || currentScores.get(p) > currentScores.get(bestScore)) {
                bestScore = p;
            }
        }

        // The finalized back trace lsit
        ArrayList<String> finalBackTrace = new ArrayList<>();
        String prevBest;

        // Traverse backwards through backTrace, at each transition adding to new in-order backtrack list finalBackTrace
        for (int i=backTrace.size()-1; i >= 0; i--) {
            prevBest = bestScore;
            finalBackTrace.add(0, prevBest);
            bestScore = backTrace.get(i).get(bestScore);
        }

        // Attach predicted POS to each word observation and print to output
        String rline = "";
        for (int i = 0; i < spline.length; i++) {
            rline += spline[i] + "/" + finalBackTrace.get(i) + " ";
        }
        rline += "\n";
        return rline;
    }

    /**
     * Runs the viterbi algorithm with a given sentence file and tag file; outputs total, correct, and wrong # of tags
     * @param sentenceFile, file name of sentence file
     * @param tagFile, file name of tag file
     * @throws IOException
     */
    public void viterbiFileTest(String sentenceFile, String tagFile) throws IOException {
        BufferedReader sr = new BufferedReader(new FileReader(sentenceFile));
        BufferedReader tr = new BufferedReader(new FileReader(tagFile));

        String sline;
        String[] tline;

        int wrong = 0;
        int correct = 0;
        int total = 0;

        // Reads through file and runs viterbi on every sentence
        while ((sline = sr.readLine()) != null) {
            tline = tr.readLine().toLowerCase().split(" ");
            sline = viterbi(sline);
            String[] spline = sline.split(" ");
            for (int i=0; i < spline.length - 1; i++) {
                total++;

                String[] pos = spline[i].split("/");
                if (pos[1].equals(tline[i])) {
                    correct++;
                }
                else wrong++;
            }
        }
        System.out.println("Viterbi File Test:");
        System.out.println("Sentence File: " + sentenceFile);
        System.out.println("Tag File: " + tagFile);
        System.out.println("# Total tags: " + total);
        System.out.println("# Correct: " + correct);
        System.out.println("# Wrong: " + wrong);
    }

    /**
     * Runs the viterbi algorithm through the console; outputs total, correct, and wrong # of tags as well as the
     * sentence and tags themselves
     * @throws Exception
     */
    public void viterbiConsoleTest() throws Exception {
        // Continues running until user quits
        while (true) {
            System.out.println("Viterbi Console Test:");
            String[] spline;
            String[] tpline;
            String[] vspline;
            String vline; // Holds the line produced by viterbi (word/POS format)
            String sline; // Holds the line of tags inputted by the user
            String tline; // Holds the line of words inputted by the user

            int total = 0;
            int correct = 0;
            int wrong = 0;

            // Console
            System.out.println("Write a test sentence, or 'q' to quit: ");
            Scanner in = new Scanner(System.in);
            sline = in.nextLine().toLowerCase();

            // Quitting
            if (sline.equals("q")) {
                break;
            }

            vline = viterbi(sline);
            spline = sline.split(" ");

            System.out.println("Input each word's corresponding POS: ");
            tline = in.nextLine().toLowerCase();
            tpline = tline.split(" ");

            // Exception
            if (tpline.length != spline.length) {
                throw new Exception("Please input valid corresponding tag line.");
            }

            // Iterates through the sentence and checks how many tags are correct/wrong
            vspline = vline.split(" ");
            for (int i = 0; i < spline.length; i++) {
                total++;

                String[] pos = vspline[i].split("/");
                if (pos[1].equals(tpline[i])) {
                    correct++;
                } else wrong++;
            }

            // Output
            System.out.println("Input Sentence: " + sline);
            System.out.println("Input Tags: " + tline);
            System.out.println("Viterbi generated line: " + vline);
            System.out.println("# Total tags: " + total);
            System.out.println("# Correct: " + correct);
            System.out.println("# Wrong: " + wrong);
        }
    }

    public static void main(String[] args) throws Exception {
        POS p = new POS();
        p.trainTables();
        p.viterbiFileTest("PS5/our-test-sentences.txt", "PS5/our-test-tags.txt");
        System.out.println();
        p.viterbiConsoleTest();
    }
}