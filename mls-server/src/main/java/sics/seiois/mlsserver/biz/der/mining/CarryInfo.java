package sics.seiois.mlsserver.biz.der.mining;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;

// the information carried by ree_next
public class CarryInfo implements Serializable {
    private static final long serialVersionUID = 2270495887417040428L;

    int attribute_nonoverlap;
    int predicate_nonoverlap;
    double attribute_distance;
    HashMap<Integer, HashSet<Integer>> tupleIDs_cons_t0;  // <pid, tid>.
    HashMap<Integer, HashSet<Integer>> tupleIDs_cons_t1;  // <pid, tid>.
    HashMap<String, HashMap<Integer, HashSet<Integer>>> tupleIDs_nonCons;// <'pid_pid', tid_t0, {tid_t1}>
    long tuple_coverage;

    public CarryInfo(int attribute_nonoverlap, int predicate_nonoverlap, double attribute_distance,
                     HashMap<Integer, HashSet<Integer>> tupleIDs_cons_t0,
                     HashMap<Integer, HashSet<Integer>> tupleIDs_cons_t1,
                     HashMap<String, HashMap<Integer, HashSet<Integer>>> tupleIDs_nonCons,
                     long tuple_coverage) {
        this.attribute_nonoverlap = attribute_nonoverlap;
        this.predicate_nonoverlap = predicate_nonoverlap;
        this.attribute_distance = attribute_distance;
        this.tupleIDs_cons_t0 = tupleIDs_cons_t0;
        this.tupleIDs_cons_t1 = tupleIDs_cons_t1;
        this.tupleIDs_nonCons = tupleIDs_nonCons;
        this.tuple_coverage = tuple_coverage;
    }

    public CarryInfo(HashMap<Integer, HashSet<Integer>> tupleIDs_cons_t0,
                     HashMap<Integer, HashSet<Integer>> tupleIDs_cons_t1) {
        this.tupleIDs_cons_t0 = tupleIDs_cons_t0;
        this.tupleIDs_cons_t1 = tupleIDs_cons_t1;
    }

    public int getAttribute_nonoverlap() {
        return this.attribute_nonoverlap;
    }

    public int getPredicate_nonoverlap() {
        return this.predicate_nonoverlap;
    }

    public double getAttribute_distance() {
        return this.attribute_distance;
    }

    public HashMap<Integer, HashSet<Integer>> getTupleIDs_cons_t0() {
        return this.tupleIDs_cons_t0;
    }

    public HashMap<Integer, HashSet<Integer>> getTupleIDs_cons_t1() {
        return this.tupleIDs_cons_t1;
    }

    public HashMap<String, HashMap<Integer, HashSet<Integer>>> getTupleIDs_nonCons() {
        return this.tupleIDs_nonCons;
    }

    public long getTuple_coverage() {
        return this.tuple_coverage;
    }
}
