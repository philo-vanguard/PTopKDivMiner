package sics.seiois.mlsserver.biz.der.mining.utils;


import sics.seiois.mlsserver.biz.der.bitset.IBitSet;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;
import sics.seiois.mlsserver.biz.der.mining.model.MLPFilterClassifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 * broadcast lattice data
 */

public class BroadcastLattice implements Serializable {
    private static final long serialVersionUID = 7770770360229647717L;

    private List<Predicate> allPredicates;
    private HashSet<IBitSet> invalidX;
    private HashMap<IBitSet, ArrayList<Predicate>> invalidRHSs;
    private HashMap<IBitSet, ArrayList<Predicate>> validXRHSs;
    private Interestingness interestingness;
    private String topKOption;
    private double KthScore;
    private HashMap<PredicateSet, Double> suppRatios;
    private PredicateProviderIndex predicateProviderIndex;
    private String option;

    // for RL
    private int ifRL = 0;
    private int ifOnlineTrainRL;
    private int ifOfflineTrainStage;
    private String PI_path;
    private String RL_code_path;
    private boolean ifExistModel;
    private float learning_rate;
    private float reward_decay;
    private float e_greedy;
    private int replace_target_iter;
    private int memory_size;
    private int batch_size;

    private ArrayList<Predicate> allExistPredicates;

    private String table_name;
    private int N;

    private MLPFilterClassifier dqnmlp;
    private boolean ifDQN;
    private HashMap<String, Integer> predicatesHashIDs;

    HashMap<IBitSet, HashSet<Predicate>> invalidXRHSs;
    HashMap<IBitSet, HashSet<Predicate>> validREEsMap;
    HashMap<Integer, HashMap<IBitSet, HashSet<Predicate>>> validLatticeVertexMap;

    private HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> Q_next;
    private boolean ifPrune;

    private boolean if_top_k;

    private int iteration;


    public BroadcastLattice(List<Predicate> allPredicates, HashSet<IBitSet> invalidX,
                            HashMap<IBitSet, HashSet<Predicate>> invalidXRHSs,
                            HashMap<IBitSet, HashSet<Predicate>> validREEsMap,
                            HashMap<Integer, HashMap<IBitSet, HashSet<Predicate>>> validLatticeVertexMap,
                            PredicateProviderIndex predicateProviderIndex,
                            HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> Q_next,
                            boolean ifPrune, boolean if_top_k, int iteration) {
        this.allPredicates = allPredicates;
        this.invalidX = invalidX;
        this.invalidXRHSs = invalidXRHSs;
        this.validREEsMap = validREEsMap;
        this.validLatticeVertexMap = validLatticeVertexMap;
        this.predicateProviderIndex = predicateProviderIndex;
        this.Q_next = Q_next;
        this.ifPrune = ifPrune;
        this.if_top_k = if_top_k;
        this.iteration = iteration;
    }


    public List<Predicate> getAllPredicates () {
        return this.allPredicates;
    }

    public ArrayList<Predicate> getAllExistPredicates() {
        return this.allExistPredicates;
    }

    public HashSet<IBitSet> getInvalidX() {
        return this.invalidX;
    }

    public HashMap<IBitSet, ArrayList<Predicate>> getInvalidRHSs() {
        return this.invalidRHSs;
    }

    public HashMap<IBitSet, ArrayList<Predicate>> getValidXRHSs() {
        return this.validXRHSs;
    }

    public double getKthScore() {
        return this.KthScore;
    }

    public Interestingness getInterestingness() {
        return interestingness;
    }

    public HashMap<PredicateSet, Double> getSuppRatios() {
        return this.suppRatios;
    }

    public PredicateProviderIndex getPredicateProviderIndex() {
        return this.predicateProviderIndex;
    }

    public String getOption() {
        return this.option;
    }

    public int getIfRL() {
        return this.ifRL;
    }

    public int getIfOnlineTrainRL() {
        return this.ifOnlineTrainRL;
    }

    public int getIfOfflineTrainStage() {
        return this.ifOfflineTrainStage;
    }

    public boolean getIfExistModel() {
        return this.ifExistModel;
    }

    public String getPI_path() {
        return this.PI_path;
    }

    public String getRL_code_path() {
        return this.RL_code_path;
    }

    public float getLearning_rate() {
        return this.learning_rate;
    }

    public float getReward_decay() {
        return this.reward_decay;
    }

    public float getE_greedy() {
        return this.e_greedy;
    }

    public int getReplace_target_iter() {
        return this.replace_target_iter;
    }

    public int getMemory_size() {
        return this.memory_size;
    }

    public int getBatch_size() {
        return this.batch_size;
    }

    public String getTable_name() {
        return this.table_name;
    }

    public int getN() {
        return this.N;
    }

    public MLPFilterClassifier getDqnmlp() {
        return this.dqnmlp;
    }

    public boolean getIfDQN() {
        return this.ifDQN;
    }

    public HashMap<String, Integer> getPredicatesHashIDs() {
        return this.predicatesHashIDs;
    }

    public String getTopKOption() {
        return this.topKOption;
    }

    public HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> getQ_next() {
        return this.Q_next;
    }

    public boolean isIfPrune() {
        return this.ifPrune;
    }

    public boolean isIf_top_k() {
        return this.if_top_k;
    }

    public int getIteration() {
        return this.iteration;
    }

    public HashMap<IBitSet, HashSet<Predicate>> getInvalidXRHSs() {
        return this.invalidXRHSs;
    }

    public HashMap<IBitSet, HashSet<Predicate>> getValidREEsMap() {
        return this.validREEsMap;
    }

    public HashMap<Integer, HashMap<IBitSet, HashSet<Predicate>>> getValidLatticeVertexMap() {
        return this.validLatticeVertexMap;
    }

}
