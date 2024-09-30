package sics.seiois.mlsserver.biz.der.mining;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.bitset.IBitSet;
import sics.seiois.mlsserver.biz.der.metanome.input.Input;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;
import sics.seiois.mlsserver.biz.der.mining.utils.Diversity;
import sics.seiois.mlsserver.biz.der.mining.utils.Relevance;

import java.io.Serializable;
import java.util.*;

public class Message implements Serializable {
    private static final long serialVersionUID = -7007743277256247070L;
    private static final Logger logger = LoggerFactory.getLogger(Message.class);
    private PredicateSet current;
    private PredicateSet rhsList;
    private PredicateSet validRHSs;
    private PredicateSet invalidRHSs; // RHS predicates that do not satisfy the support

//    private ArrayList<Predicate> intermediateRHSs; // RHS that satisfy support but not satisfy confidence
//    private ArrayList<Long> intermediateSupports; // supports of intermediate rules
    private HashMap<Integer, Long> intermediateRHSsSupport; // <rhsID, support>
    private HashMap<Integer, Double> intermediateRHSsConfidence; // <rhsID, confidence>

    private PredicateSet candidateRHSs; // RHS predicates that X can be further expanded
    private ArrayList<Long> candidateSupports;

    private long currentSupp;
    private long currentSuppCP0;
    private long currentSuppCP1;
    // support and confidence for each valid RHS, current -> p_0, p_0 \in validRHSs
    private ArrayList<Double> confidences;
    private ArrayList<Long> supports;

    // only for efficiency
    transient private PredicateSet currentSet = null;

    // only for heavy skew partition method
    HashMap<Integer, Long> allCurrentRHSsSupport;

    // for other diversity measures. Record the marginal gain on diversity.
    HashMap<Integer, Integer> attribute_nonoverlap; // <predicate_id, attr_nonoverlap>.   for different RHS
    HashMap<Integer, Integer> predicate_nonoverlap; // <predicate_id, pred_nonoverlap>.   for different RHS
    HashMap<Integer, Double> attribute_distance;    // <predicate_id, attr_distance>.     for different RHS
    ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t0; // <rhsID, pid, tid>.
    ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t1; // <rhsID, pid, tid>.
    ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>> tupleIDs_nonCons; // <rhsID, 'pid_pid', tid_t0, {tid_t1}>
    HashMap<Integer, Long> marginal_tuple_coverage;         // <predicate_id, tuple_coverage>.    for different RHS

    boolean compute_div_only = false;

    boolean empty = false;


    public HashMap<Predicate, Long> getAllCurrentRHSsSupport() {
        HashMap<Predicate, Long> map = new HashMap<>();
        for (Map.Entry<Integer, Long> entry : this.allCurrentRHSsSupport.entrySet()) {
            Integer i = entry.getKey();
            Predicate predicate = PredicateSet.getPredicate(i);
            map.put(predicate,entry.getValue());
        }

        return map;
    }

    public HashMap<Integer, Long> getAllCurrentRHSsSupportOrigin() {
        return this.allCurrentRHSsSupport;
    }

    public void transformCurrent() {
        this.currentSet = new PredicateSet();
        for (Predicate p : current) {
            this.currentSet.add(p);
        }
    }

    public PredicateSet getCurrentSet() {
        return this.current;
    }

    public PredicateSet getRHSList() {
        return this.rhsList;
    }

    public HashMap<Integer, Integer> getAttribute_nonoverlap() {
        return this.attribute_nonoverlap;
    }

    public int getAttribute_nonoverlap(int pred_id) {
        return this.attribute_nonoverlap.get(pred_id);
    }

    public HashMap<Integer, Integer> getPredicate_nonoverlap() {
        return this.predicate_nonoverlap;
    }

    public int getPredicate_nonoverlap(int pred_id) {
        return this.predicate_nonoverlap.get(pred_id);
    }

    public HashMap<Integer, Long> getMarginalTupleCoverage() {
        return this.marginal_tuple_coverage;
    }

    public long getMarginalTupleCoverage(int pred_id) {
        return this.marginal_tuple_coverage.get(pred_id);
    }

    public HashMap<Integer, Double> getAttribute_distance() {
        return this.attribute_distance;
    }

    public double getAttribute_distance(int pred_id) {
        return this.attribute_distance.get(pred_id);
    }

    public ArrayList<HashMap<Integer, HashSet<Integer>>> getTupleIDs_cons_t0() {
        return this.tupleIDs_cons_t0;
    }

    public ArrayList<HashMap<Integer, HashSet<Integer>>> getTupleIDs_cons_t1() {
        return this.tupleIDs_cons_t1;
    }

    public ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>> getTupleIDs_nonCons() {
        return this.tupleIDs_nonCons;
    }

    public void updateAllCurrentRHSsSupport(Long[] counts, ArrayList<Predicate> p_arr) {
        for (int i = 0; i < p_arr.size(); i++) {
            Integer p = PredicateSet.getIndex(p_arr.get(i));
            if (this.allCurrentRHSsSupport.containsKey(p)) {
                this.allCurrentRHSsSupport.put(p, this.allCurrentRHSsSupport.get(p) + counts[i]);
            } else {
                this.allCurrentRHSsSupport.put(p, counts[i]);
            }
        }
    }

    public void updateOneCurrentRHSsSupport(Long count, Predicate rhs) {
        Integer rhsId = PredicateSet.getIndex(rhs);
        if (this.allCurrentRHSsSupport.containsKey(rhsId)) {
            this.allCurrentRHSsSupport.put(rhsId, this.allCurrentRHSsSupport.get(rhsId) + count);
        } else {
            this.allCurrentRHSsSupport.put(rhsId, count);
        }
    }

    public void updateAllCurrentRHSsSupport(HashSet<Predicate> p_arr) {
        for (Predicate predicate : p_arr) {
            Integer p = PredicateSet.getIndex(predicate);
            if (! this.allCurrentRHSsSupport.containsKey(p)) {
                this.allCurrentRHSsSupport.put(p, 0L);
            }
        }
    }

    public void mergeAllCurrentRHSsSupport(HashMap<Predicate, Long> allCurrentRHSsSupport_) {
        for (Map.Entry<Predicate, Long> entry : allCurrentRHSsSupport_.entrySet()) {
//            logger.info(">>>>rhs: {} | support: {}", entry.getKey(), entry.getValue());

            Integer key = PredicateSet.getIndex(entry.getKey());
            if (this.allCurrentRHSsSupport.containsKey(key)) {
                this.allCurrentRHSsSupport.put(key, this.allCurrentRHSsSupport.get(key) + entry.getValue());
            } else {
                this.allCurrentRHSsSupport.put(key, entry.getValue());
            }
//            logger.info(">>>>support become: {}", allCurrentRHSsSupport.get(key));
        }
    }

    // get the information for work units with the maximum score
    public CarryInfo getCarriedInfoMax(Predicate rhs) {
        int rhs_index = 0;
        for (Predicate p : this.rhsList) {
            if (p.equals(rhs)) {
                break;
            }
            rhs_index++;
        }
        int rhs_bit_index = PredicateSet.getIndex(rhs);

        int info_attribute_nonoverlap = 0;
        int info_predicate_nonoverlap = 0;
        double info_attribute_distance = 0.0;
        HashMap<Integer, HashSet<Integer>> info_tupleIDs_cons_t0 = null;
        HashMap<Integer, HashSet<Integer>> info_tupleIDs_cons_t1 = null;
        HashMap<String, HashMap<Integer, HashSet<Integer>>> info_tupleIDs_nonCons = null;
        long info_tuple_coverage = 0L;
        
        if (!this.attribute_nonoverlap.isEmpty()) {
            info_attribute_nonoverlap = this.attribute_nonoverlap.get(rhs_bit_index);
        }
        
        if (!this.predicate_nonoverlap.isEmpty()) {
            info_predicate_nonoverlap = this.predicate_nonoverlap.get(rhs_bit_index);
        }
        
        if (!this.attribute_distance.isEmpty()) {
            info_attribute_distance = this.attribute_distance.get(rhs_bit_index);
        }
        
        if (!this.tupleIDs_cons_t0.isEmpty()) {
            info_tupleIDs_cons_t0 = this.tupleIDs_cons_t0.get(rhs_index);
        }
        
        if (!this.tupleIDs_cons_t1.isEmpty()) {
            info_tupleIDs_cons_t1 = this.tupleIDs_cons_t1.get(rhs_index);
        }
        
        if (!this.tupleIDs_nonCons.isEmpty()) {
            info_tupleIDs_nonCons = this.tupleIDs_nonCons.get(rhs_index);
        }
        
        if (!this.marginal_tuple_coverage.isEmpty()) {
            info_tuple_coverage = this.marginal_tuple_coverage.get(rhs_bit_index);
        }

        CarryInfo info_ = new CarryInfo(info_attribute_nonoverlap,
                info_predicate_nonoverlap, info_attribute_distance,
                info_tupleIDs_cons_t0, info_tupleIDs_cons_t1, info_tupleIDs_nonCons,
                info_tuple_coverage);

        return info_;

    }

    public CarryInfo getCarriedInfoLight(Predicate rhs) {
        int rhs_index = 0;
        boolean findIndex = false;
        for (Predicate p : this.rhsList) {
            if (p.equals(rhs)) {
                findIndex = true;
                break;
            }
            rhs_index++;
        }
        if (!findIndex) {
            logger.info("#### getCarriedInfoLight: not find rhs_index");
            return null;
        }

//        logger.info("#### rhs_index: {}", rhs_index);

        HashMap<Integer, HashSet<Integer>> info_tupleIDs_cons_t0 = null;
        HashMap<Integer, HashSet<Integer>> info_tupleIDs_cons_t1 = null;
//        HashMap<String, HashMap<Integer, HashSet<Integer>>> info_tupleIDs_nonCons = null;

        if (!this.tupleIDs_cons_t0.isEmpty()) {
            info_tupleIDs_cons_t0 = this.tupleIDs_cons_t0.get(rhs_index);
        }

        if (!this.tupleIDs_cons_t1.isEmpty()) {
            info_tupleIDs_cons_t1 = this.tupleIDs_cons_t1.get(rhs_index);
        }

//        if (!this.tupleIDs_nonCons.isEmpty()) {
//            info_tupleIDs_nonCons = this.tupleIDs_nonCons.get(rhs_index);
//        }

        CarryInfo info_ = new CarryInfo(info_tupleIDs_cons_t0, info_tupleIDs_cons_t1);

        return info_;

    }

    public void mergeMessage(Message message) {
        this.currentSupp += message.getCurrentSupp();
        this.currentSuppCP0 += message.getCurrentSuppCP0();
        this.currentSuppCP1 += message.getCurrentSuppCP1();
//        // set invalid RHSs
//        for (Predicate rhs : message.getInvalidRHSs()) {
//            this.invalidRHSs.add(rhs);
//        }
//        // set valid RHSs
//        for (Predicate rhs : message.getValidRHSs()) {
//            this.validRHSs.add(rhs);
//        }
        this.mergeAllCurrentRHSsSupport(message.getAllCurrentRHSsSupport());

    }

    public void mergeMessage(Message message, ArrayList<String> rel_funcs, ArrayList<String> div_funcs) {
//        if (rel_funcs.contains("support") || rel_funcs.contains("confidence") || div_funcs.contains("tuple_coverage")) {
            this.mergeMessage(message);
//        }
        if (div_funcs.contains("tuple_coverage")) {
            this.mergeTupleIDs(message);
            for (Map.Entry<Integer, Long> entry : message.getMarginalTupleCoverage().entrySet()) {
                int pred_id = entry.getKey();
                this.marginal_tuple_coverage.putIfAbsent(pred_id, 0L);
                this.marginal_tuple_coverage.put(pred_id, this.marginal_tuple_coverage.get(pred_id) + entry.getValue());
            }
        }
        if (div_funcs.contains("attribute_nonoverlap")) {
            if (!message.getAttribute_nonoverlap().isEmpty()) {
                this.attribute_nonoverlap = message.getAttribute_nonoverlap();
            }
        }
        if (div_funcs.contains("predicate_nonoverlap")) {
            if (!message.getPredicate_nonoverlap().isEmpty()) {
                this.predicate_nonoverlap = message.getPredicate_nonoverlap();
            }
        }
        if (div_funcs.contains("attribute_distance")) {
            if (!message.getAttribute_distance().isEmpty()) {
                this.attribute_distance = message.getAttribute_distance();
            }
        }
    }

    public void mergeTupleIDs(Message message) {
        ArrayList<HashMap<Integer, HashSet<Integer>>> msg_tupleIDs_cons_t0 = message.getTupleIDs_cons_t0(); // <rhsID, pid, tid>.
        ArrayList<HashMap<Integer, HashSet<Integer>>> msg_tupleIDs_cons_t1 = message.getTupleIDs_cons_t1(); // <rhsID, pid, tid>.
//        ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>> msg_tupleIDs_nonCons = message.getTupleIDs_nonCons(); // <rhsID, 'pid_pid', tid_t0, {tid_t1}>

        for (int rhsID = 0; rhsID < this.tupleIDs_cons_t0.size(); rhsID++) {
            // merge tupleIDs_cons_t0
            if (!msg_tupleIDs_cons_t0.isEmpty()) {
                for (Map.Entry<Integer, HashSet<Integer>> entry_msg : msg_tupleIDs_cons_t0.get(rhsID).entrySet()) {
                    int pid = entry_msg.getKey();
                    this.tupleIDs_cons_t0.get(rhsID).putIfAbsent(pid, new HashSet<>());
                    this.tupleIDs_cons_t0.get(rhsID).get(pid).addAll(entry_msg.getValue());
                }
            }

            // merge tupleIDs_cons_t1
            if (!msg_tupleIDs_cons_t1.isEmpty()) {
                for (Map.Entry<Integer, HashSet<Integer>> entry_msg : msg_tupleIDs_cons_t1.get(rhsID).entrySet()) {
                    int pid = entry_msg.getKey();
                    this.tupleIDs_cons_t1.get(rhsID).putIfAbsent(pid, new HashSet<>());
                    this.tupleIDs_cons_t1.get(rhsID).get(pid).addAll(entry_msg.getValue());
                }
            }

            // merge tupleIDs_nonCons
            /*
            if (!msg_tupleIDs_nonCons.isEmpty()) {
                for (Map.Entry<String, HashMap<Integer, HashSet<Integer>>> entry_msg : msg_tupleIDs_nonCons.get(rhsID).entrySet()) {
                    String key = entry_msg.getKey(); // 'pid_pid'
                    this.tupleIDs_nonCons.get(rhsID).putIfAbsent(key, new HashMap<>());
                    for (Map.Entry<Integer, HashSet<Integer>> entry_msg_tids : entry_msg.getValue().entrySet()) {
                        int tid_t0 = entry_msg_tids.getKey();
                        this.tupleIDs_nonCons.get(rhsID).get(key).putIfAbsent(tid_t0, new HashSet<>());
                        this.tupleIDs_nonCons.get(rhsID).get(key).get(tid_t0).addAll(entry_msg_tids.getValue());
                    }
                }
            }
             */
        }
    }

    public void mergeMessageWithOtherMeasures(List<Message> messages) {
        for (Message msg : messages) {
            if (!msg.getCurrentSet().equals(this.current) || !msg.getRHSList().equals(this.rhsList)) {
                continue;
            }
            if (!msg.getAttribute_nonoverlap().isEmpty()) {
                this.attribute_nonoverlap = msg.getAttribute_nonoverlap();
            }
            if (!msg.getPredicate_nonoverlap().isEmpty()) {
                this.predicate_nonoverlap = msg.getPredicate_nonoverlap();
            }
            if (!msg.getAttribute_distance().isEmpty()) {
                this.attribute_distance = msg.getAttribute_distance();
            }
        }
    }

    public void initializeMarginalDiv(String attr_pred) {
        if (attr_pred.equals("attribute_nonoverlap")) {
            for (Predicate p : this.rhsList) {
                this.attribute_nonoverlap.put(PredicateSet.getIndex(p), 0);
            }
        }
        else if (attr_pred.equals("predicate_nonoverlap")) {
            for (Predicate p : this.rhsList) {
                this.predicate_nonoverlap.put(PredicateSet.getIndex(p), 0);
            }
        }
        else if (attr_pred.equals("attribute_distance")) {
            for (Predicate p : this.rhsList) {
                this.attribute_distance.put(PredicateSet.getIndex(p), 0.0);
            }
        }
    }

    public void updateMessage_old(long support, double confidence, long maxTupleRelation) {
//        logger.info(">>>>update show rhs: {}", this.allCurrentRHSsSupport);
        for (Map.Entry<Integer, Long> entry : this.allCurrentRHSsSupport.entrySet()) {
            Predicate rhs = PredicateSet.getPredicate(entry.getKey());
            long supportXRHS = entry.getValue();
            if (rhs.isConstant()) {
                // t1 constant RHSs have been removed when obtaining applicationRHSs
//                if (rhs.getIndex1() == 1) {
//                    continue;
//                }
//                logger.info(">>>> {} support : {} * {} | {}", rhs, supportXRHS, maxTupleRelation, support);
                if (supportXRHS * maxTupleRelation < support) {
//                    logger.info(">>>> {} support : {} * {} < {}", rhs, supportXRHS, maxTupleRelation, support);
                    this.addInValidRHS(rhs);
                } else {
                    double conf;
                    if (rhs.getIndex1() == 0) {
                        conf = supportXRHS * 1.0 / this.currentSuppCP0;
                    } else {
                        conf = supportXRHS * 1.0 / this.currentSuppCP1; // no use
                    }
//                    logger.info(">>>> {} conf : {} | {}", rhs, conf, confidence);
                    if (conf >= confidence) {
                        this.addValidRHS(rhs, supportXRHS * maxTupleRelation, conf);
                    } else {
                        this.addIntermediateResults(rhs, supportXRHS * maxTupleRelation, conf);
                    }
                }
            } else {
//                logger.info(">>>> {} support : {} | {}", rhs, supportXRHS, support);
                if (supportXRHS < support) {
//                    logger.info(">>>> {} support : {} < {}", rhs, supportXRHS, support);
                    this.addInValidRHS(rhs);
                } else {
                    double conf = supportXRHS * 1.0 / this.currentSupp;
//                    logger.info(">>>> {} conf : {} | {}", rhs, conf, confidence);
                    if (conf >= confidence) {
                        this.addValidRHS(rhs, supportXRHS, conf);
                    } else {
                        this.addIntermediateResults(rhs, supportXRHS, conf);
                    }
                }
            }
        }
    }

    public void updateMessage(long support, double confidence, long maxTupleRelation) {
//        logger.info(">>>>update show rhs: {}", this.allCurrentRHSsSupport);
        for (Map.Entry<Integer, Long> entry : this.allCurrentRHSsSupport.entrySet()) {
            Predicate rhs = PredicateSet.getPredicate(entry.getKey());
            long supportXRHS = entry.getValue();
//            if (rhs.isConstant()) {
//                // t1 constant RHSs have been removed when obtaining applicationRHSs
////                if (rhs.getIndex1() == 1) {
////                    continue;
////                }
//                if (rhs.getIndex1() == 0) {
//                    supportXRHS = supportXRHS * this.currentSuppCP1; // wrong. should sum(|t0|*|t1|) in each hypercube block
//                } else {  // no use
//                    supportXRHS = supportXRHS * this.currentSuppCP0; // wrong. should sum(|t0|*|t1|) in each hypercube block
//                }
//            }
            if (supportXRHS < support) {
//                logger.info(">>>> {} support : {} < {}", rhs, supportXRHS, support);
                this.addInValidRHS(rhs);
            } else {
                double conf = supportXRHS * 1.0 / this.currentSupp;
//                logger.info(">>>> {} conf : {} | {}", rhs, conf, confidence);
                if (conf >= confidence) {
                    this.addValidRHS(rhs, supportXRHS, conf);
                } else {
                    this.addIntermediateResults(rhs, supportXRHS, conf);
                }
            }
        }
    }

    @Override
    public String toString() {
        return "[ Current : " + current.toString() + " ] " + " [ valid : " + validRHSs.toString() +
                " ] " + " [ invalid : " + invalidRHSs.toString() + " ] "
                + " [ attribute_nonoverlap : " + this.attribute_nonoverlap + " ] "
                + " [ predicate_nonoverlap : " + this.predicate_nonoverlap + " ] ";
    }

    public ArrayList<Long> getCandidateSupports() {
        return candidateSupports;
    }

    public ArrayList<Predicate> getCandidateRHSs() {
        return transfer(candidateRHSs);
    }

    public PredicateSet getCandidateRHSsOrigin() {
        return candidateRHSs;
    }

    public void setCurrentSupp(long currentSupp) {
        this.currentSupp = currentSupp;
    }

    public long getCurrentSupp() {
        return this.currentSupp;
    }

    public long getCurrentSuppCP0() {
        return this.currentSuppCP0;
    }

    public long getCurrentSuppCP1() {
        return this.currentSuppCP1;
    }

    public ArrayList<Double> getConfidences() {
        return this.confidences;
    }

    public ArrayList<Long> getSupports() {
        return this.supports;
    }

    public boolean isEmpty() {
        return this.empty;
    }

    public Message() {
        this.current = new PredicateSet();
        this.rhsList = new PredicateSet();
        this.validRHSs = new PredicateSet();
        this.invalidRHSs = new PredicateSet();
        this.supports = new ArrayList<>();
        this.confidences = new ArrayList<>();
        this.candidateRHSs = new PredicateSet();
        this.candidateSupports = new ArrayList<>();
        this.currentSupp = 0L;
        this.currentSuppCP0 = 0L;
        this.currentSuppCP1 = 0L;
        this.allCurrentRHSsSupport = new HashMap<>();
        this.intermediateRHSsSupport = new HashMap<>();
        this.intermediateRHSsConfidence = new HashMap<>();

        this.attribute_nonoverlap = new HashMap<>();
        this.predicate_nonoverlap = new HashMap<>();
        this.attribute_distance = new HashMap<>();

        this.tupleIDs_cons_t0 = new ArrayList<>();
        this.tupleIDs_cons_t1 = new ArrayList<>();
        this.tupleIDs_nonCons = new ArrayList<>();
        this.marginal_tuple_coverage = new HashMap<>();

        this.empty = true;
    }

    public Message(ArrayList<Predicate> current, long support, long supportCP0, long supportCP1) {
        PredicateSet ps = new PredicateSet();
        for (Predicate predicate : current) {
            ps.add(predicate);
        }

        this.current = ps;
        this.validRHSs = new PredicateSet();
        this.invalidRHSs = new PredicateSet();
        this.supports = new ArrayList<>();
        this.confidences = new ArrayList<>();
        this.candidateRHSs = new PredicateSet();
        this.candidateSupports = new ArrayList<>();
        this.currentSupp = support;
        this.currentSuppCP0 = supportCP0;
        this.currentSuppCP1 = supportCP1;

        this.allCurrentRHSsSupport = new HashMap<>();
        this.intermediateRHSsSupport = new HashMap<>();
        this.intermediateRHSsConfidence = new HashMap<>();
    }

    public Message(ArrayList<Predicate> current, ArrayList<Predicate> rhsList, long support, long supportCP0, long supportCP1,
                   ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t0,
                   ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t1,
                   ArrayList<HashMap<String, HashMap<Integer, HashSet<Integer>>>> tupleIDs_nonCons,
                   Long[] marginal_tuple_coverage) {
        this(current, support, supportCP0, supportCP1);
        PredicateSet ps = new PredicateSet();
        for (Predicate predicate : rhsList) {
            ps.add(predicate);
        }
        this.rhsList = ps;
        this.tupleIDs_cons_t0 = tupleIDs_cons_t0;
        this.tupleIDs_cons_t1 = tupleIDs_cons_t1;
        this.tupleIDs_nonCons = tupleIDs_nonCons;

        this.marginal_tuple_coverage = new HashMap<>();
        for (int i = 0; i < rhsList.size(); i++) {
            this.marginal_tuple_coverage.put(PredicateSet.getIndex(rhsList.get(i)), marginal_tuple_coverage[i]);
        }

        this.attribute_nonoverlap = new HashMap<>();
        this.predicate_nonoverlap = new HashMap<>();
        this.attribute_distance = new HashMap<>();
    }

    public Message(ArrayList<Predicate> current, ArrayList<Predicate> rhsList,long support, long supportCP0, long supportCP1,
                   Long[] marginal_tuple_coverage,
                   ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t0,
                   ArrayList<HashMap<Integer, HashSet<Integer>>> tupleIDs_cons_t1) {
        this(current, support, supportCP0, supportCP1);
        PredicateSet ps = new PredicateSet();
        for (Predicate predicate : rhsList) {
            ps.add(predicate);
        }
        this.rhsList = ps;

        this.marginal_tuple_coverage = new HashMap<>();
        if (marginal_tuple_coverage.length > 0) {
            for (int i = 0; i < rhsList.size(); i++) {
                this.marginal_tuple_coverage.put(PredicateSet.getIndex(rhsList.get(i)), marginal_tuple_coverage[i]);
            }
        }

        this.tupleIDs_cons_t0 = tupleIDs_cons_t0;
        this.tupleIDs_cons_t1 = tupleIDs_cons_t1;
        this.tupleIDs_nonCons = new ArrayList<>();

        this.attribute_nonoverlap = new HashMap<>();
        this.predicate_nonoverlap = new HashMap<>();
        this.attribute_distance = new HashMap<>();

        this.compute_div_only = false;
    }


    public Message(PredicateSet current, PredicateSet rhsList) {
        this.current = current;
        this.rhsList = rhsList;

        this.attribute_nonoverlap = new HashMap<>();
        this.predicate_nonoverlap = new HashMap<>();
        this.attribute_distance = new HashMap<>();
        for (Predicate rhs : this.rhsList) {
            this.attribute_nonoverlap.put(PredicateSet.getIndex(rhs), 0);
            this.predicate_nonoverlap.put(PredicateSet.getIndex(rhs), 0);
            this.attribute_distance.put(PredicateSet.getIndex(rhs), 0.0);
        }

        this.tupleIDs_cons_t0 = new ArrayList<>();
        this.tupleIDs_cons_t1 = new ArrayList<>();
        this.tupleIDs_nonCons = new ArrayList<>();
        this.marginal_tuple_coverage = new HashMap<>();

        this.validRHSs = new PredicateSet();
        this.invalidRHSs = new PredicateSet();
        this.supports = new ArrayList<>();
        this.confidences = new ArrayList<>();
        this.candidateRHSs = new PredicateSet();
        this.candidateSupports = new ArrayList<>();
        this.currentSupp = 0L;
        this.currentSuppCP0 = 0L;
        this.currentSuppCP1 = 0L;

        this.allCurrentRHSsSupport = new HashMap<>();
        this.intermediateRHSsSupport = new HashMap<>();
        this.intermediateRHSsConfidence = new HashMap<>();
    }

    public void mergeMessageBeforeSend(List<Message> messages) {
        this.compute_div_only = true;
        if (messages.isEmpty()) {
            return;
        }
        for (Message msg : messages) {
            if (!msg.getCurrentSet().equals(this.current) || !msg.getRHSList().equals(this.rhsList)) {
                continue;
            }
            this.compute_div_only = false;

            this.currentSupp = msg.getCurrentSupp();
            this.currentSuppCP0 = msg.getCurrentSuppCP0();
            this.currentSuppCP1 = msg.getCurrentSuppCP1();

            this.tupleIDs_cons_t0 = msg.getTupleIDs_cons_t0();
            this.tupleIDs_cons_t1 = msg.getTupleIDs_cons_t1();
            this.tupleIDs_nonCons = msg.getTupleIDs_nonCons();
            this.marginal_tuple_coverage = msg.getMarginalTupleCoverage();

            this.validRHSs = msg.getValidRHSsOrigin();
            this.invalidRHSs = msg.getInvalidRHSsOrigin();
            this.supports = msg.getSupports();
            this.confidences = msg.getConfidences();
            this.candidateRHSs = msg.getCandidateRHSsOrigin();
            this.candidateSupports = msg.getCandidateSupports();
            this.allCurrentRHSsSupport = msg.getAllCurrentRHSsSupportOrigin();
            this.intermediateRHSsSupport = msg.getIntermediateRHSsSupport();
            this.intermediateRHSsConfidence = msg.getIntermediateRHSsConfidence();
            break;
        }
    }

    public boolean isCompute_div_only() {
        return this.compute_div_only;
    }

    public Message(ArrayList<Predicate> current, ArrayList<Predicate> rhsList, HashMap<Integer, Integer> nonoverlap, String attr_pred) {
        PredicateSet ps1 = new PredicateSet();
        for (Predicate predicate : current) {
            ps1.add(predicate);
        }
        this.current = ps1;

        PredicateSet ps2 = new PredicateSet();
        for (Predicate predicate : rhsList) {
            ps2.add(predicate);
        }
        this.rhsList = ps2;

        this.attribute_nonoverlap = new HashMap<>();
        this.predicate_nonoverlap = new HashMap<>();
        this.attribute_distance = new HashMap<>();

        if (attr_pred.equals("attr_nonoverlap")) {
            this.attribute_nonoverlap = nonoverlap;
        } else if (attr_pred.equals("pred_nonoverlap")) {
            this.predicate_nonoverlap = nonoverlap;
        }

        this.tupleIDs_cons_t0 = new ArrayList<>();
        this.tupleIDs_cons_t1 = new ArrayList<>();
        this.tupleIDs_nonCons = new ArrayList<>();
        this.marginal_tuple_coverage = new HashMap<>();

        this.validRHSs = new PredicateSet();
        this.invalidRHSs = new PredicateSet();
        this.supports = new ArrayList<>();
        this.confidences = new ArrayList<>();
        this.candidateRHSs = new PredicateSet();
        this.candidateSupports = new ArrayList<>();
        this.currentSupp = 0L;
        this.currentSuppCP0 = 0L;
        this.currentSuppCP1 = 0L;

        this.allCurrentRHSsSupport = new HashMap<>();
        this.intermediateRHSsSupport = new HashMap<>();
        this.intermediateRHSsConfidence = new HashMap<>();
    }

    public Message(ArrayList<Predicate> current, ArrayList<Predicate> rhsList, HashMap<Integer, Double> attribute_distance) {
        PredicateSet ps1 = new PredicateSet();
        for (Predicate predicate : current) {
            ps1.add(predicate);
        }
        this.current = ps1;

        PredicateSet ps2 = new PredicateSet();
        for (Predicate predicate : rhsList) {
            ps2.add(predicate);
        }
        this.rhsList = ps2;

        this.attribute_nonoverlap = new HashMap<>();
        this.predicate_nonoverlap = new HashMap<>();
        this.attribute_distance = attribute_distance;

        this.tupleIDs_cons_t0 = new ArrayList<>();
        this.tupleIDs_cons_t1 = new ArrayList<>();
        this.tupleIDs_nonCons = new ArrayList<>();
        this.marginal_tuple_coverage = new HashMap<>();

        this.validRHSs = new PredicateSet();
        this.invalidRHSs = new PredicateSet();
        this.supports = new ArrayList<>();
        this.confidences = new ArrayList<>();
        this.candidateRHSs = new PredicateSet();
        this.candidateSupports = new ArrayList<>();
        this.currentSupp = 0L;
        this.currentSuppCP0 = 0L;
        this.currentSuppCP1 = 0L;

        this.allCurrentRHSsSupport = new HashMap<>();
        this.intermediateRHSsSupport = new HashMap<>();
        this.intermediateRHSsConfidence = new HashMap<>();
    }

    public void addValidRHS(Predicate p, long supp, double conf) {
        this.validRHSs.add(p);
        this.supports.add(supp);
        this.confidences.add(conf);
    }

    public void addCandidateRHS(Predicate p, long candSupp) {
        this.candidateRHSs.add(p);
        this.candidateSupports.add(candSupp);
    }

    public void addInValidRHS(Predicate p) {
        this.invalidRHSs.add(p);
    }

    public ArrayList<Predicate> getValidRHSs() {
        return transfer(this.validRHSs);
    }

    public PredicateSet getValidRHSsOrigin() {
        return this.validRHSs;
    }

    public ArrayList<Predicate> getInvalidRHSs() {
        return transfer(this.invalidRHSs);
    }

    public PredicateSet getInvalidRHSsOrigin() {
        return this.invalidRHSs;
    }

    public void addIntermediateResults(Predicate p, long supp, double conf) {
        this.intermediateRHSsSupport.put(PredicateSet.getIndex(p), supp);
        this.intermediateRHSsConfidence.put(PredicateSet.getIndex(p), conf);
    }

    public HashMap<Integer, Long> getIntermediateRHSsSupport() {
        return this.intermediateRHSsSupport;
    }

    public HashMap<Integer, Double> getIntermediateRHSsConfidence() {
        return this.intermediateRHSsConfidence;
    }

    public double getIntermediateRHSsConfidence(int p_id) {
        return this.intermediateRHSsConfidence.get(p_id);
    }

    public ArrayList<Predicate> getCurrent() {
        return transfer(this.current);
    }

    private ArrayList<Predicate> transfer(PredicateSet predicates){
        ArrayList<Predicate> ret = new ArrayList<>();
        for (Predicate predicate : predicates) {
            ret.add(predicate);
        }
        return ret;
    }

}
