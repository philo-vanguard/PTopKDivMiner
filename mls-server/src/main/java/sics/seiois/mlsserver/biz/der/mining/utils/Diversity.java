package sics.seiois.mlsserver.biz.der.mining.utils;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.metanome.REE;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;
import sics.seiois.mlsserver.biz.der.mining.CarryInfo;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;


public class Diversity implements Serializable {
    private static final long serialVersionUID = 7726501371439798509L;
    private static Logger logger = LoggerFactory.getLogger(Diversity.class);

    private ArrayList<String> candidate_functions;

    // the attributes covered by current partial solution Sigma
    private HashSet<String> covered_attributes;

    // the predicates covered by current partial solution Sigma
    private HashSet<Predicate> covered_predicates;

    // min distance between rule pairs in current partial solution
    private double min_distance;

    // the tuple covered by current partial solution Sigma
    // for constant RHS
    private HashMap<Integer, HashSet<Integer>> covered_tids_cons_t0; // <pid, <tids>>
    private HashMap<Integer, HashSet<Integer>> covered_tids_cons_t1; // <pid, <tids>>
    // for non-constant RHS
    private HashMap<String, HashMap<Integer, HashSet<Integer>>> covered_tids_nonCons;  // <'pid_pid', tid_t0, {tid_t1}>


    public Diversity() {
        this.candidate_functions = new ArrayList<>();

        this.covered_attributes = new HashSet<>();
        this.covered_predicates = new HashSet<>();

        this.covered_tids_cons_t0 = new HashMap<>();
        this.covered_tids_cons_t1 = new HashMap<>();
        this.covered_tids_nonCons = new HashMap<>();

        this.candidate_functions.add("attribute_nonoverlap");
        this.candidate_functions.add("predicate_nonoverlap");
        this.candidate_functions.add("tuple_coverage");
        this.candidate_functions.add("attribute_distance");

        this.min_distance = Double.POSITIVE_INFINITY;
    }

    public Diversity(String diversity) {
        this.candidate_functions = new ArrayList<>();

        this.covered_attributes = new HashSet<>();
        this.covered_predicates = new HashSet<>();

        this.covered_tids_cons_t0 = new HashMap<>();
        this.covered_tids_cons_t1 = new HashMap<>();
        this.covered_tids_nonCons = new HashMap<>();

        this.min_distance = Double.POSITIVE_INFINITY;

        if (diversity.equals("null")) {
            return;
        }

        for (String func : diversity.split("#")) {
            this.candidate_functions.add(func);
        }

        logger.info("#### diversity measures: {}", this.candidate_functions);
    }

    public ArrayList<String> getCandidate_functions() {
        return this.candidate_functions;
    }

    public HashSet<String> getCoveredAttributes() {
        return this.covered_attributes;
    }

    public HashSet<Predicate> getCoveredPredicates() {
        return this.covered_predicates;
    }

    public HashSet<Integer> getCoveredTidsCons_t0_by_Sigma(int pid) {
        return this.covered_tids_cons_t0.get(pid);
    }

    public boolean isContain_CoveredTidsCons_t0_by_Sigma(int pid) {
        return this.covered_tids_cons_t0.get(pid) != null;
    }

    public boolean isEmpty_CoveredTidsCons_t0_by_Sigma() {
        return this.covered_tids_cons_t0.isEmpty();
    }

    public HashSet<Integer> getCoveredTidsCons_t1_by_Sigma(int pid) {
        return this.covered_tids_cons_t1.get(pid);
    }

    public boolean isContain_CoveredTidsCons_t1_by_Sigma(int pid) {
        return this.covered_tids_cons_t1.get(pid) != null;
    }

    public boolean isEmpty_CoveredTidsCons_t1_by_Sigma() {
        return this.covered_tids_cons_t1.isEmpty();
    }

    public HashMap<String, HashMap<Integer, HashSet<Integer>>> getCoveredTids_nonCons() {
        return this.covered_tids_nonCons;
    }

    public double getMin_distance() {
        return this.min_distance;
    }

    // update the diversity information covered by the current partial solution
    public void updateCurrentPartialSolutionDiv(REE ree_next, CarryInfo info_next, ArrayList<REE> partial_solution) {
        PredicateSet Xset = ree_next.getCurrentList();
        Predicate rhs = ree_next.getRHS();
        for (String div : this.candidate_functions) {
            switch (div) {
                // update covered attributes
                case "attribute_nonoverlap":
                    for (Predicate p : Xset) {
                        this.covered_attributes.add("X_" + p.getOperand1().getColumnLight().getName());
                        if (!p.isConstant()) {
                            this.covered_attributes.add("X_" + p.getOperand2().getColumnLight().getName());
                        }
                    }
                    this.covered_attributes.add("Y_" + rhs.getOperand1().getColumnLight().getName());
                    if (!rhs.isConstant()) {
                        this.covered_attributes.add("Y_" + rhs.getOperand2().getColumnLight().getName());
                    }
                    break;

                // update covered predicates
                case "predicate_nonoverlap":
                    for (Predicate p : Xset) {
                        this.covered_predicates.add(p);
                    }
                    this.covered_predicates.add(rhs);
                    break;

                // update covered constant tuple ids
                case "tuple_coverage":
                    if (info_next != null) {
                        this.updateTupleIDs(info_next);
                    }
                    break;

                // update min distance in current partial solution
                case "attribute_distance":
                    for (REE ree : partial_solution) {
                        double dis = ree_next.computeAttributeJaccardDistance(ree);
                        if (dis < this.min_distance) {
                            this.min_distance = dis;
                        }
                    }
                    break;
            }
        }
    }

    public void updateCurrentPartialSolutionDiv(CarryInfo info_next) {
        // update covered constant tuple ids
        if (info_next != null) {
            this.updateTupleIDs(info_next);
        }
    }

    public void updateTupleIDs(CarryInfo info) {
        HashMap<Integer, HashSet<Integer>> info_tupleIDs_cons_t0 = info.getTupleIDs_cons_t0(); // <pid, tid>.
        HashMap<Integer, HashSet<Integer>> info_tupleIDs_cons_t1 = info.getTupleIDs_cons_t1(); // <pid, tid>.
//        HashMap<String, HashMap<Integer, HashSet<Integer>>> info_tupleIDs_nonCons = info.getTupleIDs_nonCons(); // <'pid_pid', tid_t0, {tid_t1}>

        // update covered_tids_cons_t0
        if (info_tupleIDs_cons_t0 != null && !info_tupleIDs_cons_t0.isEmpty()) {
            for (Map.Entry<Integer, HashSet<Integer>> entry : info_tupleIDs_cons_t0.entrySet()) {
                int pid = entry.getKey();
                this.covered_tids_cons_t0.putIfAbsent(pid, new HashSet<>());
                this.covered_tids_cons_t0.get(pid).addAll(entry.getValue());
            }
        }

        // update covered_tids_cons_t1
        if (info_tupleIDs_cons_t1 != null && !info_tupleIDs_cons_t1.isEmpty()) {
            for (Map.Entry<Integer, HashSet<Integer>> entry : info_tupleIDs_cons_t1.entrySet()) {
                int pid = entry.getKey();
                this.covered_tids_cons_t1.putIfAbsent(pid, new HashSet<>());
                this.covered_tids_cons_t1.get(pid).addAll(entry.getValue());
            }
        }

        // update covered_tids_nonCons
        /*
        if (info_tupleIDs_nonCons != null && !info_tupleIDs_nonCons.isEmpty()) {
            for (Map.Entry<String, HashMap<Integer, HashSet<Integer>>> entry : info_tupleIDs_nonCons.entrySet()) {
                String key = entry.getKey(); // 'pid_pid'
                this.covered_tids_nonCons.putIfAbsent(key, new HashMap<>());
                for (Map.Entry<Integer, HashSet<Integer>> entry_tids : entry.getValue().entrySet()) {
                    int tid_t0 = entry_tids.getKey();
                    this.covered_tids_nonCons.get(key).putIfAbsent(tid_t0, new HashSet<>());
                    this.covered_tids_nonCons.get(key).get(tid_t0).addAll(entry_tids.getValue());
                }
            }
        }
        */
    }

    public double getDivScore(REE ree, ArrayList<REE> partial_solution, int MAX_CURRENT_PREDICTES,
                              double w_attr_non, double w_pred_non, double w_attr_dis, double w_tuple_cov) {
        PredicateSet Xset = ree.getCurrentList();
        Predicate rhs = ree.getRHS();
        double diversity_score = 0.0;
        for (String div : this.candidate_functions) {
            switch (div) {
                case "attribute_nonoverlap":
                    // attributes in ree
                    HashSet<String> attrs_in_ree = new HashSet<>();
                    for (Predicate p : Xset) {
                        attrs_in_ree.add("X_" + p.getOperand1().getColumnLight().getName());
                        if (!p.isConstant()) {
                            attrs_in_ree.add("X_" + p.getOperand2().getColumnLight().getName());
                        }
                    }
                    attrs_in_ree.add("Y_" + rhs.getOperand1().getColumnLight().getName());
                    if (!rhs.isConstant()) {
                        attrs_in_ree.add("Y_" + rhs.getOperand2().getColumnLight().getName());
                    }
                    // intersection
                    HashSet<String> intersection_result = new HashSet<>();
                    intersection_result.addAll(this.covered_attributes);
                    intersection_result.retainAll(attrs_in_ree);
                    // compute attribute nonoverlap score
                    double div_attr_score = -intersection_result.size() * 1.0 / MAX_CURRENT_PREDICTES;
                    diversity_score += w_attr_non * div_attr_score;
                    break;

                case "predicate_nonoverlap":
                    // predicates in ree
                    HashSet<Predicate> pred_in_ree = new HashSet<>();
                    for (Predicate p : Xset) {
                        pred_in_ree.add(p);
                    }
                    pred_in_ree.add(rhs);
                    // intersection
                    HashSet<Predicate> inter_result = new HashSet<>();
                    inter_result.addAll(this.covered_predicates);
                    inter_result.retainAll(pred_in_ree);
                    // compute predicate nonoverlap score
                    double div_pred_score = -inter_result.size() * 1.0 / MAX_CURRENT_PREDICTES;
                    diversity_score += w_pred_non * div_pred_score;
                    break;

                case "tuple_coverage":
                    logger.info("#### For tuple_coverage measure, information is re-computed for all work units." +
                            "In each iteration, there should return a non-null ree_next, so it will not find the next ree from validREEs.");
                    System.exit(1);
                    break;

                case "attribute_distance":
                    double min_dis = Double.POSITIVE_INFINITY;
                    for (REE ree_discovered : partial_solution) {
                        double dis = ree.computeAttributeJaccardDistance(ree_discovered);
                        if (dis < min_dis) {
                            min_dis = dis;
                        }
                    }
                    if (min_dis < this.min_distance) {
                        min_dis = this.min_distance;
                    }
                    double marginal_dis = min_dis - this.min_distance;
                    diversity_score += w_attr_dis * marginal_dis;
                    break;
            }
        }
        return diversity_score;
    }


    // print information of the current partial solution
    public void printInfo() {
        logger.info("#### For the current partial solution, covered_attributes: {}", this.covered_attributes);
        logger.info("#### For the current partial solution, covered_predicates: {}", this.covered_predicates);
        logger.info("#### For the current partial solution, min_attribute_dis: {}", this.min_distance);

        long covered_cons_t0_size = 0L;
        for (Map.Entry<Integer, HashSet<Integer>> entry : this.covered_tids_cons_t0.entrySet()) {
            covered_cons_t0_size += entry.getValue().size();
        }
        logger.info("#### For the current partial solution, covered_constantRHS_t0 size: {}", covered_cons_t0_size);

        long covered_cons_t1_size = 0L;
        for (Map.Entry<Integer, HashSet<Integer>> entry : this.covered_tids_cons_t1.entrySet()) {
            covered_cons_t1_size += entry.getValue().size();
        }
        logger.info("#### For the current partial solution, covered_constantRHS_t0 size: {}", covered_cons_t1_size);
    }


    // the upper bound of an REE when \Sigma increases or when this REE is expanded further
    public double computeUB(REE ree, double w_attr_non, double w_pred_non, double w_attr_dis, double w_tuple_cov) {
        double UB = 0.0;
        for (String div : this.candidate_functions) {
            switch (div) {
                case "attribute_nonoverlap":
                    double UB_attr_non = ree.getAttr_nonoverlap_score();
                    UB += w_attr_non * UB_attr_non;
                    break;
                case "predicate_nonoverlap":
                    double UB_pred_non = ree.getPred_nonoverlap_score();
                    UB += w_pred_non * UB_pred_non;
                    break;
                case "tuple_coverage":
                    double UB_tuple_cov = ree.getTuple_cov_score();
                    UB += w_tuple_cov * UB_tuple_cov;
                    break;
                case "attribute_distance":
                    double UB_attr_dis = 0.0;
                    UB += w_attr_dis * UB_attr_dis;
                    break;
            }
        }
        return UB;
    }

    public double computeUB(HashMap<String, Double> div_score_all, double w_attr_non, double w_pred_non, double w_attr_dis, double w_tuple_cov) {
        double UB = 0.0;
        for (String div : this.candidate_functions) {
            switch (div) {
                case "attribute_nonoverlap":
                    double UB_attr_non = div_score_all.get("attribute_nonoverlap");
                    UB += w_attr_non * UB_attr_non;
                    break;
                case "predicate_nonoverlap":
                    double UB_pred_non = div_score_all.get("predicate_nonoverlap");
                    UB += w_pred_non * UB_pred_non;
                    break;
                case "tuple_coverage":
                    double UB_tuple_cov = div_score_all.get("tuple_coverage");
                    UB += w_tuple_cov * UB_tuple_cov;
                    break;
                case "attribute_distance":
                    double UB_attr_dis = 0.0;
                    UB += w_attr_dis * UB_attr_dis;
                    break;
            }
        }
        return UB;
    }

    public double computeLB(REE ree, int MAX_CURRENT_PREDICTES,
                            double w_attr_non, double w_pred_non, double w_attr_dis, double w_tuple_cov) {
        return this.computeLB(ree.getCurrentList(), ree.getRHS(), MAX_CURRENT_PREDICTES, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov);
    }

    // the lower bound of an REE when \Sigma increases
    public double computeLB(PredicateSet Xset, Predicate rhs, int MAX_CURRENT_PREDICTES,
                            double w_attr_non, double w_pred_non, double w_attr_dis, double w_tuple_cov) {
        double LB = 0.0;
        for (String div : this.candidate_functions) {
            switch (div) {
                case "attribute_nonoverlap":
                    ArrayList<String> attrs = new ArrayList<>();
                    for (Predicate p : Xset) {
                        attrs.add("X_" + p.getOperand1().getColumnLight().getName());
                        if (!p.isConstant()) {
                            attrs.add("X_" + p.getOperand2().getColumnLight().getName());
                        }
                    }
                    attrs.add("Y_" + rhs.getOperand1().getColumnLight().getName());
                    if (rhs.isConstant()) {
                        attrs.add("Y_" + rhs.getOperand2().getColumnLight().getName());
                    }
                    double LB_attr_non = -attrs.size() * 1.0 / MAX_CURRENT_PREDICTES;
                    LB += w_attr_non * LB_attr_non;
                    break;

                case "predicate_nonoverlap":
                    int pred_num = Xset.size() + 1;
                    double LB_pred_non = -pred_num * 1.0 / MAX_CURRENT_PREDICTES;
                    LB += w_pred_non * LB_pred_non;
                    break;

                case "tuple_coverage":
                    double LB_tuple_cov = 0.0;
                    LB += w_tuple_cov * LB_tuple_cov;
                    break;

                case "attribute_distance":
                    double LB_attr_dis;
                    if (this.min_distance == Double.NEGATIVE_INFINITY) { // Sigma is empty or only contains one REE.
                        LB_attr_dis = -1;
                    } else {
                        LB_attr_dis = 0.0 - this.min_distance;  // 0.0 is the minimum attribute distance between REE and Sigma, as Sigma increases.
                    }
                    LB += w_attr_dis * LB_attr_dis;
                    break;
            }
        }
        return LB;
    }
}
