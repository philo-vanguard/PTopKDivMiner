package sics.seiois.mlsserver.biz.der.mining.utils;

import gnu.trove.list.array.TIntArrayList;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.metanome.REE;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;

import java.util.*;

/*
    Now only consider REEs that only contain t_0.A = t_1.A predicates
 */


public class HyperCube {
    private static Logger logger = LoggerFactory.getLogger(HyperCube.class);
    private HashMap<TIntArrayList, HashMap<Integer, Cube>> cubeInstances;   // <x_value, <y_value, cube>>. // cube: the numbers of satisfaction for different RHS

    private int MIN_SAMPLE_SIZE = 10;
    private int MAX_SAMPLE_SIZE = 200; // 1000

    // Ys
    private ArrayList<Predicate> rhss;

    // the support of <t0, t1>
    private long supportX;
    // the support of t0
    private long supportXCP0;
    // the support of t1
    private long supportXCP1;

    // length = this.rhss.size()
    private Long[] suppRHSs;

    private Map<TIntArrayList, Integer> lhsSupportMap0;
    private Map<TIntArrayList, Integer> lhsSupportMap1;

    // for constant RHS.
    private ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t0;  // <rhsID, pid, tid>.  <partition_id, <tuple_ids>> for different RHSs;
    private ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t1;  // <rhsID, pid, tid>.  <partition_id, <tuple_ids>> for different RHSs;
    // for non-constant RHS.
    private ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>> tupleIDs_nonCons; // <rhsID, 'pid_pid', tid_t0, {tid_t1}>

    private Long[] marginal_tuple_coverage; // <rhsID, tuple_coverage>

    public HyperCube(ArrayList<Predicate> rhss_arr) {
        this.cubeInstances = new HashMap<>();
        lhsSupportMap0 = new HashMap<>();
        lhsSupportMap1 = new HashMap<>();
        /*
        this.rhss = new HashMap<>();
        for (Predicate rhs : rhss_arr) {
            if (! this.rhss.containsKey(rhs.toString())) {
                this.rhss.put(rhs.toString(), rhs);
            }
        }
         */
        this.rhss = rhss_arr;
        this.supportX = 0;
        this.supportXCP0 = 0;
        this.supportXCP1 = 0;
        this.suppRHSs = new Long[rhss_arr.size()];
        this.marginal_tuple_coverage = new Long[rhss_arr.size()];
        this.tupleIDs_cons_t0 = new ArrayList<>();
        this.tupleIDs_cons_t1 = new ArrayList<>();
        this.tupleIDs_nonCons = new ArrayList<>();
        for (int i = 0; i < this.suppRHSs.length; i++) {
            this.suppRHSs[i] = 0L;
            this.marginal_tuple_coverage[i] = 0L;
            this.tupleIDs_cons_t0.add(new HashMap<>());
            this.tupleIDs_cons_t1.add(new HashMap<>());
            this.tupleIDs_nonCons.add(new HashMap<>());
        }
    }

    /***
     *  record the satisfied tuple IDs
     ***/
    public void addTupleIDsCons(int rhsID, int pid, int tid, int index, Diversity diversity) {
        if (index == 0) {
            if (!diversity.isEmpty_CoveredTidsCons_t0_by_Sigma() &&
                    diversity.isContain_CoveredTidsCons_t0_by_Sigma(pid) &&
                    diversity.getCoveredTidsCons_t0_by_Sigma(pid).contains(tid)) {
                // remove tid that has been covered by current partial solution
                return;
            }
            this.tupleIDs_cons_t0.get(rhsID).putIfAbsent(pid, new HashSet<>());
            this.tupleIDs_cons_t0.get(rhsID).get(pid).add(tid);
        }
        else {
            if (!diversity.isEmpty_CoveredTidsCons_t1_by_Sigma() &&
                    diversity.isContain_CoveredTidsCons_t1_by_Sigma((pid)) &&
                    diversity.getCoveredTidsCons_t1_by_Sigma(pid).contains(tid)) {
                // remove tid that has been covered by current partial solution
                return;
            }
            this.tupleIDs_cons_t1.get(rhsID).putIfAbsent(pid, new HashSet<>());
            this.tupleIDs_cons_t1.get(rhsID).get(pid).add(tid);
        }
    }

    public ArrayList<HashMap<Integer, HashSet<Integer>>> getTupleIDsCons_t0() {
        return this.tupleIDs_cons_t0;
    }

    public ArrayList<HashMap<Integer, HashSet<Integer>>> getTupleIDsCons_t1() {
        return this.tupleIDs_cons_t1;
    }

    public ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>> getTupleIDs_nonCons() {
        return this.tupleIDs_nonCons;
    }

    public Long[] getMarginalTupleCoverage() {
        return this.marginal_tuple_coverage;
    }

    public void clearData() {
        this.tupleIDs_nonCons = new ArrayList<>();
        this.tupleIDs_cons_t0 = new ArrayList<>();
        this.tupleIDs_cons_t1 = new ArrayList<>();
    }

    public void addxValue0(TIntArrayList xValue){
        Integer supp = this.lhsSupportMap0.getOrDefault(xValue, 0);
        this.lhsSupportMap0.put(xValue,supp+1);
    }

    public void setxValue0(TIntArrayList xValue, int supp){
        this.lhsSupportMap0.put(xValue,supp);
    }

    public void setxValue1(TIntArrayList xValue, int supp){
        this.lhsSupportMap1.put(xValue,supp);
    }

    public void addxValue1(TIntArrayList xValue){
        Integer supp = this.lhsSupportMap1.getOrDefault(xValue, 0);
        this.lhsSupportMap1.put(xValue,supp+1);
    }

//    public void addTuple(TIntArrayList xValue, Integer yValue, int rhsID, int tid, int index) {
//        if (!this.cubeInstances.containsKey(xValue)) {
//            HashMap<Integer, Cube> tt = new HashMap<>();
//            Cube cube = new Cube();
//            cube.addTupleID(tid, rhsID, index);
//            tt.put(yValue, cube);
//            this.cubeInstances.put(xValue, tt);
//        } else {
//            HashMap<Integer, Cube> tt = this.cubeInstances.get(xValue);
//            if (!tt.containsKey(yValue)) {
//                Cube cube = new Cube();
//                cube.addTupleID(tid, rhsID, index);
//                tt.put(yValue, cube);
//            } else {
//                tt.get(yValue).addTupleID(tid, rhsID, index);
//            }
//        }
//    }

    public void addTupleT0(TIntArrayList xValue, Integer yValue, int rhsID, int tid) {
        if (!this.cubeInstances.containsKey(xValue)) {
            HashMap<Integer, Cube> tt = new HashMap<>();
            Cube cube = new Cube(suppRHSs.length); // given hash key (x_value and y_value), cube saves the number of tuple that satisfy the t0 in rule
            cube.addTupleIDT0(rhsID);
            tt.put(yValue, cube);
            this.cubeInstances.put(xValue, tt);
        } else {
            HashMap<Integer, Cube> tt = this.cubeInstances.get(xValue);
            if (!tt.containsKey(yValue)) {
                Cube cube = new Cube(suppRHSs.length);
                cube.addTupleIDT0(rhsID);
                tt.put(yValue, cube);
            } else {
                tt.get(yValue).addTupleIDT0(rhsID);
            }
        }
    }

    public void addTupleT0(TIntArrayList xValue, Integer yValue, int rhsID, int pid, int tid, Diversity diversity) {
        if (!this.cubeInstances.containsKey(xValue)) {
            HashMap<Integer, Cube> tt = new HashMap<>();
            Cube cube = new Cube(suppRHSs.length);  // given x_value and y_value, cube saves the tuple ids that satisfy the rule
            cube.addTupleIDT0(rhsID);
            if (diversity.isEmpty_CoveredTidsCons_t0_by_Sigma() ||
                    !diversity.isContain_CoveredTidsCons_t0_by_Sigma(pid) ||
                    !diversity.getCoveredTidsCons_t0_by_Sigma(pid).contains(tid)) {
                cube.addRecordTupleIDT0(rhsID, tid);  // record tuple ids
                cube.setPid_t0(pid);
            }
            tt.put(yValue, cube);
            this.cubeInstances.put(xValue, tt);
        } else {
            HashMap<Integer, Cube> tt = this.cubeInstances.get(xValue);
            if (!tt.containsKey(yValue)) {
                Cube cube = new Cube(suppRHSs.length);
                cube.addTupleIDT0(rhsID);
                if (diversity.isEmpty_CoveredTidsCons_t0_by_Sigma() ||
                        !diversity.isContain_CoveredTidsCons_t0_by_Sigma(pid) ||
                        !diversity.getCoveredTidsCons_t0_by_Sigma(pid).contains(tid)) {
                    cube.addRecordTupleIDT0(rhsID, tid);  // record tuple ids
                    cube.setPid_t0(pid);
                }
                tt.put(yValue, cube);
            } else {
                tt.get(yValue).addTupleIDT0(rhsID);
                if (diversity.isEmpty_CoveredTidsCons_t0_by_Sigma() ||
                        !diversity.isContain_CoveredTidsCons_t0_by_Sigma(pid) ||
                        !diversity.getCoveredTidsCons_t0_by_Sigma(pid).contains(tid)) {
                    tt.get(yValue).addRecordTupleIDT0(rhsID, tid);  // record tuple ids
                    tt.get(yValue).setPid_t0(pid);
                }
            }
        }
    }


    public void addTupleT1(TIntArrayList xValue, Integer yValue, int rhsID, int tid) {
        if (!this.cubeInstances.containsKey(xValue)) {
            HashMap<Integer, Cube> tt = new HashMap<>();
            Cube cube = new Cube(suppRHSs.length); // given hash key (x_value and y_value), cube saves the number of tuple that satisfy the t1 in rule
            cube.addTupleIDT1(rhsID);
            tt.put(yValue, cube);
            this.cubeInstances.put(xValue, tt);
        } else {
            HashMap<Integer, Cube> tt = this.cubeInstances.get(xValue);
            if (!tt.containsKey(yValue)) {
                Cube cube = new Cube(suppRHSs.length);
                cube.addTupleIDT1(rhsID);
                tt.put(yValue, cube);
            } else {
                tt.get(yValue).addTupleIDT1(rhsID);
            }
        }
    }

    public void addTupleT1(TIntArrayList xValue, Integer yValue, int rhsID, int pid, int tid, Diversity diversity) {
        if (!this.cubeInstances.containsKey(xValue)) {
            HashMap<Integer, Cube> tt = new HashMap<>();
            Cube cube = new Cube(suppRHSs.length); // given x_value and y_value, cube saves the tuple ids that satisfy the rule
            cube.addTupleIDT1(rhsID);
            if (diversity.isEmpty_CoveredTidsCons_t1_by_Sigma() ||
                    !diversity.getCoveredTidsCons_t1_by_Sigma(pid).contains(tid)) {
                cube.addRecordTupleIDT1(rhsID, tid);  // record tuple ids
                cube.setPid_t1(pid);
            }
            tt.put(yValue, cube);
            this.cubeInstances.put(xValue, tt);
        } else {
            HashMap<Integer, Cube> tt = this.cubeInstances.get(xValue);
            if (!tt.containsKey(yValue)) {
                Cube cube = new Cube(suppRHSs.length);
                cube.addTupleIDT1(rhsID);
                if (diversity.isEmpty_CoveredTidsCons_t1_by_Sigma() ||
                        !diversity.isContain_CoveredTidsCons_t1_by_Sigma(pid) ||
                        !diversity.getCoveredTidsCons_t1_by_Sigma(pid).contains(tid)) {
                    cube.addRecordTupleIDT1(rhsID, tid);  // record tuple ids
                    cube.setPid_t1(pid);
                }
                tt.put(yValue, cube);
            } else {
                tt.get(yValue).addTupleIDT1(rhsID);
                if (diversity.isEmpty_CoveredTidsCons_t1_by_Sigma() ||
                        !diversity.isContain_CoveredTidsCons_t1_by_Sigma(pid) ||
                        !diversity.getCoveredTidsCons_t1_by_Sigma(pid).contains(tid)) {
                    tt.get(yValue).addRecordTupleIDT1(rhsID, tid);  // record tuple ids
                    tt.get(yValue).setPid_t1(pid);
                }
            }
        }
    }

    // not right!!!
    public void getStatistic(boolean ifSame) {
        if (ifSame) {
            for (Integer value : this.lhsSupportMap0.values()) {
                long count = value;
                this.supportX += (long) (count * (count - 1));
            }
        } else {
            for (Map.Entry<TIntArrayList, Integer> entry : this.lhsSupportMap0.entrySet()) {
                if (this.lhsSupportMap1.containsKey(entry.getKey())) {
                    long count = entry.getValue();
                    long count1 = lhsSupportMap1.get(entry.getKey());
                    this.supportX += (long) (count * count1);
                }
            }
        }

        for (Map.Entry<TIntArrayList, HashMap<Integer, Cube>> entry : this.cubeInstances.entrySet()) {
            HashMap<Integer, Cube> tt = entry.getValue();
            for (Map.Entry<Integer, Cube> entry_ : tt.entrySet()) {
                Cube cube = entry_.getValue();
                if (!ifSame) {
                    for (int rhsId = 0; rhsId < this.suppRHSs.length; rhsId++) {
                        this.suppRHSs[rhsId] += (long) (cube.getTupleNumT0(rhsId) * cube.getTupleNumT1(rhsId));
                    }
                } else {
                    for (int rhsId = 0; rhsId < this.suppRHSs.length; rhsId++) {
//                        this.suppRHSs[cube.getRhsID()] += (long) (cube.getTupleNumT0() * (cube.getTupleNumT0() - 1) / 2);
                        this.suppRHSs[rhsId] += (long) (cube.getTupleNumT0(rhsId) * (cube.getTupleNumT0(rhsId) - 1));
                    }
                }
            }
        }
    }

    public void getStatistic_new(boolean ifSame, int[] pids, Diversity diversity, long maxTupleRelation,
                                 ArrayList<REE> partial_solution, double sampleRatioForTupleCov) {
        HashSet<TIntArrayList> commonList = new HashSet<>();
        // update support information of LHS
        for (Map.Entry<TIntArrayList, Integer> entry : this.lhsSupportMap0.entrySet()) {
            TIntArrayList xValue = entry.getKey();
            if (!this.lhsSupportMap1.containsKey(xValue)) {
                continue;
            }
            commonList.add(xValue);
            long count = entry.getValue();
            long count1 = this.lhsSupportMap1.get(xValue);
            this.supportX += (long) (count * count1);
            this.supportXCP0 += (long)(count);
            this.supportXCP1 += (long) (count1);
//            logger.info("#### xValue:{}, count: {}, count1: {}", xValue, count, count1);
        }

        // Note that, we do not compute this one, since the same instance for t0 and t1 DO satisfies the definition of valuation and support
//        // remove the number of (instance of t0 = instance of t1)
//        if (ifSame && pids[0]==pids[1]) {
//            this.supportX -= To_Compute;
//        }

        // for constant RHSs - update marginal tuple coverage of rules
        if (diversity.getCandidate_functions().contains("tuple_coverage")) {
            for (int rhsId = 0; rhsId < this.marginal_tuple_coverage.length; rhsId++) {
                if (!this.rhss.get(rhsId).isConstant()) {
                    continue;
                }
                for (Map.Entry<Integer, HashSet<Integer>> entry : this.tupleIDs_cons_t0.get(rhsId).entrySet()) { // <rhsID, pid, tid>
                    this.marginal_tuple_coverage[rhsId] += entry.getValue().size() * maxTupleRelation;
                }
                for (Map.Entry<Integer, HashSet<Integer>> entry : this.tupleIDs_cons_t1.get(rhsId).entrySet()) { // <rhsID, pid, tid>
                    this.marginal_tuple_coverage[rhsId] += entry.getValue().size() * maxTupleRelation;
                }
            }
        }

        // update support and tuple coverage information of rules
        for (Map.Entry<TIntArrayList, HashMap<Integer, Cube>> entry : this.cubeInstances.entrySet()) {
            TIntArrayList xValue = entry.getKey();
            if (!commonList.contains(xValue)) {
                continue;
            }
            long t0_lhs_size = this.lhsSupportMap0.get(xValue);
            long t1_lhs_size = this.lhsSupportMap1.get(xValue);
            HashMap<Integer, Cube> tt = entry.getValue();
            for (Map.Entry<Integer, Cube> entry_ : tt.entrySet()) {
                Cube cube = entry_.getValue();
                for (int rhsID = 0; rhsID < this.suppRHSs.length; rhsID++) {
                    Predicate p = this.rhss.get(rhsID);
                    // handle constant predicates as RHSs
                    if (p.isConstant()) {
                        // evaluate constant predicates and check whether these values equal to the constant
//                        if (p.getConstantInt().intValue() == entry_.getKey().intValue()) {
                            if (p.getIndex1() == 0) {
//                                this.suppRHSs[rhsID] += (long) (cube.getTupleNumT0(rhsID));  // old, will * maxOneRelationNum, which must make sure invalidX is distinguished by the type of RHS
                                this.suppRHSs[rhsID] += (long) (cube.getTupleNumT0(rhsID) * t1_lhs_size);
                            } else if (p.getIndex1() == 1) {
//                                this.suppRHSs[rhsID] += (long) (cube.getTupleNumT1(rhsID));  // old, will * maxOneRelationNum, which must make sure invalidX is distinguished by the type of RHS
                                this.suppRHSs[rhsID] += (long) (cube.getTupleNumT1(rhsID) * t0_lhs_size);
                            }
//                        }
                    }
                    // handle non-constant predicates as RHSs
                    else {
                        this.suppRHSs[rhsID] += (long) (cube.getTupleNumT0(rhsID) * cube.getTupleNumT1(rhsID));
                    }
                }

                // for non-constant RHSs, compute approximate marginal tuple coverage w.r.t. current partial solution, by sampling
                if (diversity.getCandidate_functions().contains("tuple_coverage")) {
                    // the tuple id block for t0 and t1
                    ArrayList<HashSet<Integer>> tid_t0_ = cube.getTupleIDsT0();  // <rhsID, {tid}>
                    ArrayList<HashSet<Integer>> tid_t1_ = cube.getTupleIDsT1();  // <rhsID, {tid}>
                    int pid_t0 = cube.getPid_t0();
                    int pid_t1 = cube.getPid_t1();

                    if (partial_solution.isEmpty()) {
                        for (int rhsId = 0; rhsId < this.suppRHSs.length; rhsId++) {
                            if (this.rhss.get(rhsId).isConstant()) {
                                continue;
                            }
                            int tid_t0_size = tid_t0_.get(rhsId).size();
                            int tid_t1_size = tid_t1_.get(rhsId).size();
//                            if (ifSame && pids[0]==pids[1]) {
//                                this.marginal_tuple_coverage[rhsId] += ((long) tid_t0_size * (tid_t1_size - 1));
//                            } else {
                                this.marginal_tuple_coverage[rhsId] += ((long) tid_t0_size * tid_t1_size);
//                            }
                        }
                        continue;
                    }

                    Random random = new Random();
                    for (int rhsId = 0; rhsId < this.suppRHSs.length; rhsId++) {
                        if (this.rhss.get(rhsId).isConstant()) {
                            continue;
                        }
                        List<Integer> tid_t0_list_ = new ArrayList<>(tid_t0_.get(rhsId));
                        List<Integer> tid_t1_list_ = new ArrayList<>(tid_t1_.get(rhsId));
                        HashSet<ImmutablePair<Integer, Integer>> selectedTuplePairs = new HashSet<>();
                        // randomly sample some tuple id pair
                        int tid_t0_size = tid_t0_list_.size();
                        if (tid_t0_size == 0) {
                            continue;
                        }
                        int tid_t1_size = tid_t1_list_.size();
                        if (tid_t1_size == 0) {
                            continue;
                        }
                        if (sampleRatioForTupleCov < 1.0) {
                            // logger.info("cube size: {}", tid_t0_size * tid_t1_size);
                            int sample_size;
//                        if (ifSame && pids[0]==pids[1]) {
//                            sample_size = (int) (sampleRatioForTupleCov * tid_t0_size * (tid_t1_size - 1));
//                            if (sample_size < 1) {
//                                sample_size = Math.min(tid_t0_size * (tid_t1_size - 1), MIN_SAMPLE_SIZE);
//                            }
//                        } else {
                            sample_size = (int) (sampleRatioForTupleCov * tid_t0_size * tid_t1_size);
                            if (sample_size < 1) {
                                sample_size = Math.min(tid_t0_size * tid_t1_size, MIN_SAMPLE_SIZE);
                            }
//                        }
                            if (sample_size > MAX_SAMPLE_SIZE) {
                                sample_size = MAX_SAMPLE_SIZE;
                            }
                            // compute sample_size tuple pairs
                            while (selectedTuplePairs.size() < sample_size) {
                                int random_t0 = random.nextInt(tid_t0_size);
                                int random_t1 = random.nextInt(tid_t1_size);
//                            while (ifSame && pids[0]==pids[1] && random_t0==random_t1) {
//                                random_t0 = random.nextInt(tid_t0_size);
//                                random_t1 = random.nextInt(tid_t1_size);
//                            }
                                selectedTuplePairs.add(new ImmutablePair<>(tid_t0_list_.get(random_t0), tid_t1_list_.get(random_t1)));
                            }
                        }
                        else {
                            for (Integer tid_0 : tid_t0_list_) {
                                for (Integer tid_1 : tid_t1_list_) {
                                    selectedTuplePairs.add(new ImmutablePair<>(tid_0, tid_1));
                                }
                            }
                        }

                        // check whether these tuple pairs satisfy the rules in partial solution, and count the satisfied number
                        int covered_num = this.coveredByPartialSolution(selectedTuplePairs, pid_t0, pid_t1, partial_solution);

                        // estimate the value of marginal tuple coverage, w.r.t. the current partial solution
                        double covered_ratio = covered_num * 1.0 / selectedTuplePairs.size();
                        long marginal_covered_num;
//                        if (ifSame && pids[0]==pids[1]) {
//                            marginal_covered_num = (long) ((1 - covered_ratio) * tid_t0_size * (tid_t1_size - 1));
//                        } else {
                            marginal_covered_num = (long) ((1 - covered_ratio) * tid_t0_size * tid_t1_size);
//                        }
                        this.marginal_tuple_coverage[rhsId] += marginal_covered_num;
                    }
                }

                /*
                // add satisfied tuple ids for non-constant RHSs
                if (diversity.getCandidate_functions().contains("tuple_coverage")) {
                    for (int rhsId = 0; rhsId < this.suppRHSs.length; rhsId++) {
                        // Note that, tuple pairs from <tupleIDs_t0, tupleIDs_t1> are the real match results
//                        HashMap<Integer, HashSet<Integer>> tupleIDs_t0 = cube.getTupleIDsT0(rhsId);
//                        HashMap<Integer, HashSet<Integer>> tupleIDs_t1 = cube.getTupleIDsT1(rhsId);
//                        this.tupleIDs_nonCons_t0.get(rhsId).add(tupleIDs_t0);
//                        this.tupleIDs_nonCons_t1.get(rhsId).add(tupleIDs_t1);
                        this.updateAndRemoveTidsCoveredBySigma(rhsId, cube, diversity);  // update this.tupleIDs_nonCons
                    }
                }
                */
            }
        }

        /*
        // count the number of tuple coverage
        if (diversity.getCandidate_functions().contains("tuple_coverage")) {
            for (int rhsID = 0; rhsID < this.suppRHSs.length; rhsID++) {
                long tuple_cov = 0L;
                // count tupleIDs_cons_t0 // <rhsID, pid, tid>.
                for (Map.Entry<Integer, HashSet<Integer>> entry : this.tupleIDs_cons_t0.get(rhsID).entrySet()) {
                    tuple_cov += entry.getValue().size() * maxTupleRelation;  // constant RHS
                }

                // count tupleIDs_cons_t1 // <rhsID, pid, tid>.
                for (Map.Entry<Integer, HashSet<Integer>> entry : this.tupleIDs_cons_t1.get(rhsID).entrySet()) {
                    tuple_cov += entry.getValue().size() * maxTupleRelation;  // constant RHS
                }

                // count tupleIDs_nonCons // <rhsID, 'pid_pid', tid_t0, {tid_t1}>
                for (Map.Entry<String, HashMap<Integer, HashSet<Integer>>> entry : this.tupleIDs_nonCons.get(rhsID).entrySet()) {
                    for (Map.Entry<Integer, HashSet<Integer>> entry_ : entry.getValue().entrySet()) {
                        tuple_cov += entry_.getValue().size();  // non-constant RHS
                    }
                }

                this.tuple_coverage.add(tuple_cov);
            }
        }
        */
    }


    // check whether tuple pairs satisfy the rules in current partial solution
    public int coveredByPartialSolution(HashSet<ImmutablePair<Integer, Integer>> selectedTuplePairs,
                                      int pid_t0, int pid_t1,
                                      ArrayList<REE> partial_solution) {
        int covered_num = 0;
        for (ImmutablePair<Integer, Integer> tuple_pair : selectedTuplePairs) {
            if (this.satisfyREEs(tuple_pair.getLeft(), tuple_pair.getRight(), pid_t0, pid_t1, partial_solution)) {
                covered_num++;
            }
        }
        return covered_num;
    }

    // whether tuple pair (tid0, tid1) in partitions (pid_t0, pid_t1) is covered by at least one REE in partial_solution
    public boolean satisfyREEs(int tid0, int tid1,
                               int pid_t0, int pid_t1,
                               ArrayList<REE> partial_solution) {
        boolean satisfy = false;
        for (REE ree : partial_solution) {
            boolean satisfyOne = true;
            // check X
            for (Predicate p : ree.getCurrentList()) {
                if (!this.satisfyPredicate(tid0, tid1, pid_t0, pid_t1, p)) {
                    satisfyOne = false;
                    break;
                }
            }
            if (!satisfyOne) {
                continue;
            }
            // check Y
            if (!this.satisfyPredicate(tid0, tid1, pid_t0, pid_t1, ree.getRHS())) {
                satisfyOne = false;
                continue;
            }
            satisfy = true;
            break;
        }
        return satisfy;
    }

    public boolean satisfyPredicate(int tid0, int tid1,
                                    int pid_t0, int pid_t1,
                                    Predicate predicate) {
        try {
            if (predicate.isConstant()) {
                int index = predicate.getIndex1();
                int v;
                if (index == 0) {
                    v = predicate.getOperand1().getColumnLight().getValueInt(pid_t0, tid0);
                } else {
                    v = predicate.getOperand2().getColumnLight().getValueInt(pid_t1, tid1);
                }
                if (v == predicate.getConstantInt().intValue()) {
                    return true;
                }
                return false;
            } else {
                int v0 = predicate.getOperand1().getColumnLight().getValueInt(pid_t0, tid0);
                int v1 = predicate.getOperand2().getColumnLight().getValueInt(pid_t1, tid1);
                if (v0 == v1) {  // need to check predicate.getOperator()!
                    return true;
                }
                return false;
            }
        } catch (Exception e) {
            return false;
        }
    }

    public void updateAndRemoveTidsCoveredBySigma(int rhsId, Cube cube, Diversity diversity) {
        // Note that, tuple pairs from <tupleIDs_t0, tupleIDs_t1> are the real match results
        HashSet<Integer> tupleIDs_t0 = cube.getTupleIDsT0(rhsId);
        HashSet<Integer> tupleIDs_t1 = cube.getTupleIDsT1(rhsId);
        int pid_t0 = cube.getPid_t0();
        int pid_t1 = cube.getPid_t1();

        // current tuple pairs covered by partial solution Sigma
        HashMap<String, HashMap<Integer, HashSet<Integer>>> covered_tids_nonCons_by_Sigma = diversity.getCoveredTids_nonCons(); // <'pid_pid', tid_t0, {tid_t1}>

        // update this.tupleIDs_nonCons
        String key = pid_t0 + "_" + pid_t1;
        for (Integer tid_t0 : tupleIDs_t0) {
            for (Integer tid_t1 : tupleIDs_t1) {
                this.tupleIDs_nonCons.get(rhsId).putIfAbsent(key, new HashMap<>());
                if (covered_tids_nonCons_by_Sigma.get(key) != null &&
                        covered_tids_nonCons_by_Sigma.get(key).get(tid_t0) != null &&
                        covered_tids_nonCons_by_Sigma.get(key).get(tid_t0).contains(tid_t1)) {
                    continue;
                }
                this.tupleIDs_nonCons.get(rhsId).putIfAbsent(key, new HashMap<>()); // <rhsID, 'pid_pid', tid_t0, {tid_t1}>. ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>>
                this.tupleIDs_nonCons.get(rhsId).get(key).putIfAbsent(tid_t0, new HashSet<>());
                this.tupleIDs_nonCons.get(rhsId).get(key).get(tid_t0).add(tid_t1);
            }
        }
//        cube.clearData();
    }

    public long getSupportX() {
        return this.supportX;
    }

    public long getSupportXCP0() {
        return this.supportXCP0;
    }

    public long getSupportXCP1() {
        return this.supportXCP1;
    }

    public Long[] getSuppRHSs() {
        return this.suppRHSs;
    }

    public String toSize() {
        return "cubeInstances size:" + cubeInstances.size() + " | suppRHSs size:" + suppRHSs.length;
    }

}