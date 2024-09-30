package sics.seiois.mlsserver.biz.der.mining.utils;

import sics.seiois.mlsserver.biz.der.metanome.REE;
import sics.seiois.mlsserver.biz.der.metanome.input.InputLight;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
    data to broadcast
 */
public class BroadcastObj implements Serializable {
    private static final long serialVersionUID = 1761087336068687017L;
    private int max_num_tuples;
    private InputLight inputLight;
    private long support;
    private float confidence;
    private long maxOneRelationNum;
    private HashMap<String, Long> tupleNumberRelations;
    private ArrayList<Predicate> allRealCosntantPredicates;
    private Map<PredicateSet, List<Predicate>> validConstantRule;

    private Relevance relevance;
    private Diversity diversity;
    private double sampleRatioForTupleCov;
    private ArrayList<REE> partial_solution;

    private int index_null_string;
    private int index_null_double;
    private int index_null_long;

    public BroadcastObj(int max_num_tuples, InputLight inputLight, long support,
                        float confidence, long maxOneRelationNum, HashMap<String, Long> tupleNumberRelations) {
        this.max_num_tuples = max_num_tuples;
        this.inputLight = inputLight;
        this.support = support;
        this.confidence = confidence;
        this.maxOneRelationNum = maxOneRelationNum;
        this.tupleNumberRelations = tupleNumberRelations;
    }

    public BroadcastObj(int max_num_tuples, long maxOneRelationNum, double sampleRatioForTupleCov,
                        Diversity diversity, ArrayList<REE> partial_solution,
                        int index_null_string, int index_null_double, int index_null_long) {
        this.max_num_tuples = max_num_tuples;
        this.maxOneRelationNum = maxOneRelationNum;
        this.sampleRatioForTupleCov = sampleRatioForTupleCov;
        this.diversity = diversity;
        this.partial_solution = partial_solution;
        this.index_null_string = index_null_string;
        this.index_null_double = index_null_double;
        this.index_null_long = index_null_long;
    }

    public BroadcastObj(int max_num_tuples, InputLight inputLight, long support,
                        float confidence, long maxOneRelationNum, HashMap<String, Long> tupleNumberRelations,
                        Relevance relevance, Diversity diversity, double sampleRatioForTupleCov,
                        ArrayList<REE> partial_solution,
                        int index_null_string, int index_null_double, int index_null_long) {
        this.max_num_tuples = max_num_tuples;
        this.inputLight = inputLight;
        this.support = support;
        this.confidence = confidence;
        this.maxOneRelationNum = maxOneRelationNum;
        this.tupleNumberRelations = tupleNumberRelations;
        this.relevance = relevance;
        this.diversity = diversity;
        this.sampleRatioForTupleCov = sampleRatioForTupleCov;
        this.partial_solution = partial_solution;
        this.index_null_string = index_null_string;
        this.index_null_double = index_null_double;
        this.index_null_long = index_null_long;
    }

    public BroadcastObj(int max_num_tuples, InputLight inputLight, long support,
                        float confidence, long maxOneRelationNum, HashMap<String, Long> tupleNumberRelations,
                        ArrayList<Predicate> allRealCosntantPredicates) {
        this.max_num_tuples = max_num_tuples;
        this.inputLight = inputLight;
        this.support = support;
        this.confidence = confidence;
        this.maxOneRelationNum = maxOneRelationNum;
        this.tupleNumberRelations = tupleNumberRelations;
        this.allRealCosntantPredicates = allRealCosntantPredicates;
    }

    public Relevance getRelevance() {
        return this.relevance;
    }

    public Diversity getDiversity() {
        return this.diversity;
    }

    public double getSampleRatioForTupleCov() {
        return this.sampleRatioForTupleCov;
    }

    public ArrayList<REE> getPartial_solution() {
        return this.partial_solution;
    }

    public int getMax_num_tuples() {
        return this.max_num_tuples;
    }

    public InputLight getInputLight() {
        return this.inputLight;
    }

    public long getSupport() {
        return this.support;
    }

    public float getConfidence() {
        return this.confidence;
    }

    public long getMaxOneRelationNum() {
        return this.maxOneRelationNum;
    }

    public HashMap<String, Long> getTupleNumberRelations() {
        return this.tupleNumberRelations;
    }

    public ArrayList<Predicate> getAllRealCosntantPredicates() {
        return this.allRealCosntantPredicates;
    }

    public Map<PredicateSet, List<Predicate>> getValidConstantRule() {
        return this.validConstantRule;
    }

    public void setValidConstantRule(Map<PredicateSet, List<Predicate>> validConstantRule) {
        this.validConstantRule = validConstantRule;
    }

    public int getIndex_null_double() {
        return this.index_null_double;
    }

    public int getIndex_null_long() {
        return this.index_null_long;
    }

    public int getIndex_null_string() {
        return this.index_null_string;
    }

}
