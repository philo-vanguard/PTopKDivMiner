package sics.seiois.mlsserver.biz.der.mining.utils;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.Path;
import org.ejml.simple.SimpleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;


public class RelevanceModel implements Serializable {
    private static Logger logger = LoggerFactory.getLogger(RelevanceModel.class);

    private SimpleMatrix predicateEmbedMatrix;           // shape (predicate_size, predicate_embedding_size)
    private SimpleMatrix weightPredicates;               // shape (1, predicate_embedding_size)
    private SimpleMatrix weightREEsEmbed;                // shape (predicate_embedding_size * 2, rees_embedding_size)
    private SimpleMatrix weightRelevance;                // shape (ree_embedding_size, 1)
    private SimpleMatrix weightUBSub;                    // shape (1, 1)

    static public int MAX_LHS_PREDICATES = 5;
    static public int MAX_RHS_PREDICATES = 1;
    static public String PADDING_VALUE = "PAD";

    HashMap<String, Integer> predicateToIDs;

    public RelevanceModel(String predicateToIDFile, String relevanceModelFile, FileSystem hdfs) {
        try {
            this.loadPredicateIDs(predicateToIDFile, hdfs);
            this.loadRelevanceModel(relevanceModelFile, hdfs);
        } catch (Exception e) {

        }
    }

    public void loadRelevanceModel(String relevanceModelFile, FileSystem hdfs) throws Exception {
        FSDataInputStream inputTxt = hdfs.open(new Path(relevanceModelFile));
        BufferedInputStream bis = new BufferedInputStream(inputTxt);
        InputStreamReader sReader = new InputStreamReader(bis, "UTF-8");
        BufferedReader bReader = new BufferedReader(sReader);
        ArrayList<SimpleMatrix> matrices = new ArrayList<>();
        String line = null;
        boolean beginMatrix = true;
        while ( (line = bReader.readLine()) != null) {
            if (line.trim().equals("")) {
                continue;
            }
            int row = 0, col = 0;
            if (beginMatrix) {
                String[] info = line.split(" ");
                row = Integer.parseInt(info[0].trim());
                col = Integer.parseInt(info[1].trim());
            }
            double[][] mat = new double[row][col];
            for (int r = 0; r < row; r++) {
                line = bReader.readLine();
                String[] info = line.split(" ");
                for (int c = 0; c < info.length; c++) {
                    mat[r][c] = Double.parseDouble(info[c].trim());
                }
            }
            SimpleMatrix matrix = new SimpleMatrix(mat);
            matrices.add(matrix);
        }

        // assign to matrices
        this.predicateEmbedMatrix = matrices.get(0);
        this.weightPredicates = matrices.get(1);
        this.weightREEsEmbed = matrices.get(2);
        this.weightRelevance = matrices.get(3);
        this.weightUBSub = matrices.get(4);

//        logger.info("predicateEmbedMatrix size: {}*{}", predicateEmbedMatrix.numRows(), predicateEmbedMatrix.numCols());
//        logger.info("weightPredicates size: {}*{}", weightPredicates.numRows(), weightPredicates.numCols());
//        logger.info("weightREEsEmbed size: {}*{}", weightREEsEmbed.numRows(), weightREEsEmbed.numCols());
//        logger.info("weightRelevance size: {}*{}", weightRelevance.numRows(), weightRelevance.numCols());
//        logger.info("weightUBSub size: {}*{}", weightUBSub.numRows(), weightUBSub.numCols());
//
//        logger.info("predicateEmbedMatrix:\n{}", predicateEmbedMatrix);
//        logger.info("weightPredicates:\n{}", weightPredicates);
//        logger.info("weightREEsEmbed:\n{}", weightREEsEmbed);
//        logger.info("weightRelevance:\n{}", weightRelevance);
//        logger.info("weightUBSub:\n{}", weightUBSub);

//        // weights of features are non-negative, old version
//        this.weightUBSubj = this.weightUBSubj.elementMult(this.weightUBSubj);
    }

    private void loadPredicateIDs(String predicateToIDFile, FileSystem hdfs) throws Exception {
        FSDataInputStream inputTxt = hdfs.open(new Path(predicateToIDFile));
        BufferedInputStream bis = new BufferedInputStream(inputTxt);
        InputStreamReader sReader = new InputStreamReader(bis, "UTF-8");
        BufferedReader bReader = new BufferedReader(sReader);

        this.predicateToIDs = new HashMap<>();
        String line = null;
        int ID = 0;
//        logger.info("#### predicateToIDs:");
        while ((line = bReader.readLine()) != null) {
            if (line.trim().equals("")) {
                continue;
            }
            String predicate_ = line.trim();
            this.predicateToIDs.put(predicate_, ID);
//            logger.info("#### predicate: {}, ID: {}", predicate_, ID);
            ID++;
        }
    }

    public double predict_score(PredicateSet Xset, Predicate rhs) {
        ArrayList<Predicate> reeLHS = new ArrayList<>();
        for (Predicate p : Xset) {
            reeLHS.add(p);
        }
        ArrayList<Predicate> reeRHS = new ArrayList<>();
        reeRHS.add(rhs);
        int[] ree_lhs = this.transformFeature(reeLHS, MAX_LHS_PREDICATES);
        int[] ree_rhs = this.transformFeature(reeRHS, MAX_RHS_PREDICATES);
        // 1. get embeddings of all predicate embeddings
        ArrayList<SimpleMatrix> predicateEmbedsLHS = this.extractEmbeddings(ree_lhs);
        ArrayList<SimpleMatrix> predicateEmbedsRHS = this.extractEmbeddings(ree_rhs);
        // 2. get embeddings of LHS and RHS
        SimpleMatrix lhsEmbed = this.clauseEmbed(predicateEmbedsLHS);
        SimpleMatrix rhsEmbed = this.clauseEmbed(predicateEmbedsRHS);
        // 3. get embeddings of REE
        SimpleMatrix reeEmbed = this.concatTwoByCol(lhsEmbed, rhsEmbed);
        reeEmbed = this.ReLU(reeEmbed.mult(this.weightREEsEmbed));
        // 4. compute the rule relevance, i.e., the subjective score
        SimpleMatrix subjectiveScore = reeEmbed.mult(this.weightRelevance);
        // 5. compute the rule relevance subjective score with UB
        subjectiveScore = this.weightUBSub.minus(this.ReLU(subjectiveScore));
        return subjectiveScore.get(0, 0);
    }

    private int[] transformFeature(ArrayList<Predicate> selectedPredicates, int max_num_predicates) {
        int[] features = new int[max_num_predicates];
        int count = 0;
        for (Predicate p : selectedPredicates) {
//            logger.info("#### predicate: {}", p.toString().trim());
            features[count++] = this.predicateToIDs.get(p.toString().trim());
        }
        for (int i = count; i < features.length; i++) {
            features[i] = this.predicateToIDs.get(PADDING_VALUE);
        }
        return features;
    }

    private ArrayList<SimpleMatrix> extractEmbeddings(int[] predicateIDs) {
        ArrayList<SimpleMatrix> res = new ArrayList<>();
        for (int i = 0; i < predicateIDs.length; i++) {
            int tid = predicateIDs[i];
            res.add(this.predicateEmbedMatrix.extractVector(true, tid)); // <predicate_num, 1, predicate_embedding_dim>
        }
        return res;
    }

    private SimpleMatrix ReLU(SimpleMatrix input) {
        SimpleMatrix output = new SimpleMatrix(input);
        for (int i = 0; i < output.numRows(); ++i) {
            for (int j = 0; j < output.numCols(); ++j) {
                output.set(i, j, Math.max(0, output.get(i, j)));
            }
        }
        return output;
    }

    /*
    private SimpleMatrix clauseEmbed(ArrayList<SimpleMatrix> embeddingsC) {
        SimpleMatrix res = embeddingsC.get(0).elementMult(this.weightPredicates.extractVector(true, 0));
        for (int i = 1; i < embeddingsC.size(); i++) {
            res = res.plus(embeddingsC.get(i).elementMult(this.weightPredicates.extractVector(true, 0)));
        }
        res = res.divide(embeddingsC.size() * 1.0); // plus & divide -> reduce_mean in Python
        res = this.ReLU(res);
        return res;
    }
    */

    private SimpleMatrix clauseEmbed(ArrayList<SimpleMatrix> embeddingsC) {
        ArrayList<SimpleMatrix> results = new ArrayList<>();
        SimpleMatrix weight_predicate = this.weightPredicates.extractVector(true, 0);
        for (SimpleMatrix pred_embed_mat : embeddingsC) {
            SimpleMatrix res = this.ReLU(pred_embed_mat.elementMult(weight_predicate));
            results.add(res);
        }
        // plus & divide -> reduce_mean in Python
        SimpleMatrix res = results.get(0);
        for (int i = 1; i < results.size(); i++) {
            res = res.plus(results.get(i));
        }
        res = res.divide(results.size() * 1.0);
        return res;

//        SimpleMatrix res = this.ReLU(embeddingsC.get(0).elementMult(this.weightPredicates.extractVector(true, 0)));
//        for (int i = 1; i < embeddingsC.size(); i++) {
//            res = res.plus(this.ReLU(embeddingsC.get(i).elementMult(this.weightPredicates.extractVector(true, 0))));
//        }
//        res = res.divide(embeddingsC.size() * 1.0); // plus & divide -> reduce_mean in Python
//        return res;
    }

    private SimpleMatrix concatTwoByCol(SimpleMatrix one, SimpleMatrix two) {
        int row = one.numRows();
        int col = one.numCols() + two.numCols();
        SimpleMatrix res = new SimpleMatrix(row, col);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < one.numCols(); c++) {
                res.set(r, c, one.get(r, c));
            }
            for (int c = 0; c < two.numCols(); c++) {
                res.set(r, c + one.numCols(), two.get(r, c));
            }
        }
        return res;
    }

    public double getWeightUBSubj() {
        return this.weightUBSub.get(0, 0);
    }

}
