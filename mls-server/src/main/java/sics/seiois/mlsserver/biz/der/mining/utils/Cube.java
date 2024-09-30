package sics.seiois.mlsserver.biz.der.mining.utils;

/*
    Now only consider REEs that only contain t_0.A = t_1.A predicates
 */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class Cube {
//    private Map<Integer, Map<Integer, Integer>> tupleIDs;
    private long[] tupleIDs_t0;
    private long[] tupleIDs_t1;

    private long[] tuple_coverage;

    private ArrayList<HashSet<Integer>> record_tupleIDs_t0;  // <rhsID, {tid}>.  tuple_ids for different RHSs.
    private ArrayList<HashSet<Integer>> record_tupleIDs_t1;  // <rhsID, {tid}>.  tuple_ids for different RHSs.
    private int pid_t0;
    private int pid_t1;


    public Cube(int size) {
        this.tupleIDs_t0 = new long[size];
        this.tupleIDs_t1 = new long[size];
        this.tuple_coverage = new long[size];
        this.record_tupleIDs_t0 = new ArrayList<>();
        this.record_tupleIDs_t1 = new ArrayList<>();
        for (int index = 0; index < size; index++) {
            this.tupleIDs_t0[index] = 0;
            this.tupleIDs_t1[index] = 0;
            this.tuple_coverage[index] = 0;
            this.record_tupleIDs_t0.add(new HashSet<>());
            this.record_tupleIDs_t1.add(new HashSet<>());
        }
    }

//    public void addTupleID(int rhsId, int index) {
//        Map<Integer, Integer> indexTupleIDs = tupleIDs.get(index);
//        Integer t = indexTupleIDs.getOrDefault(rhsId, 0);
//        indexTupleIDs.put(rhsId, t + 1);
//    }
//
//    public long getTupleNum(int rhsId, int index) {
//        Map<Integer, Integer> indexTupleIDs = this.tupleIDs.get(index);
//        return indexTupleIDs.getOrDefault(rhsId, 0);
//    }


    /***
     *  record the satisfied tuple IDs
     ***/
    public void addRecordTupleIDT0(int rhsID, int tid) {
        this.record_tupleIDs_t0.get(rhsID).add(tid);
    }

    public void setPid_t0(int pid_t0) {
        this.pid_t0 = pid_t0;
    }

    public int getPid_t0() {
        return this.pid_t0;
    }

    public void addRecordTupleIDT1(int rhsID, int tid) {
        this.record_tupleIDs_t1.get(rhsID).add(tid);
    }

    public void setPid_t1(int pid_t1) {
        this.pid_t1 = pid_t1;
    }

    public int getPid_t1() {
        return this.pid_t1;
    }

    public HashSet<Integer> getTupleIDsT0(int rhsID) {
        return this.record_tupleIDs_t0.get(rhsID);
    }

    public ArrayList<HashSet<Integer>> getTupleIDsT0() {
        return this.record_tupleIDs_t0;
    }

    public HashSet<Integer> getTupleIDsT1(int rhsID) {
        return this.record_tupleIDs_t1.get(rhsID);
    }

    public ArrayList<HashSet<Integer>> getTupleIDsT1() {
        return this.record_tupleIDs_t1;
    }

    public void clearData() {
        this.record_tupleIDs_t0 = new ArrayList<>();
        this.record_tupleIDs_t1 = new ArrayList<>();
    }


    /***
     *  record the number of satisfied tuples
     ***/
    public void addTupleIDT0(int rhsId) {
//        Integer t = tupleIDs_t0.getOrDefault(rhsId, 0);
//        tupleIDs_t0.put(rhsId, t + 1);
        tupleIDs_t0[rhsId] = tupleIDs_t0[rhsId] + 1;
    }

    public void addTupleIDT1(int rhsId) {
//        Integer t = tupleIDs_t1.getOrDefault(rhsId, 0);
//        tupleIDs_t1.put(rhsId, t + 1);
        tupleIDs_t1[rhsId] = tupleIDs_t1[rhsId] + 1;
    }

    public long getTupleNumT0(int rhsId) {
//        return this.tupleIDs_t0.getOrDefault(rhsId, 0);
        return tupleIDs_t0[rhsId];
    }

    public long getTupleNumT1(int rhsId) {
//        return this.tupleIDs_t1.getOrDefault(rhsId, 0);
        return tupleIDs_t1[rhsId];
    }

}