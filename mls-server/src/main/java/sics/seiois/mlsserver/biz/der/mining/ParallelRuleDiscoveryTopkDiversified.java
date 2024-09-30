package sics.seiois.mlsserver.biz.der.mining;


import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.bitset.IBitSet;
import sics.seiois.mlsserver.biz.der.metanome.REE;
import sics.seiois.mlsserver.biz.der.metanome.input.Input;
import sics.seiois.mlsserver.biz.der.metanome.input.InputLight;
import sics.seiois.mlsserver.biz.der.metanome.input.ParsedColumn;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;
import sics.seiois.mlsserver.biz.der.mining.utils.*;
import sics.seiois.mlsserver.model.SparkContextConfig;
import sics.seiois.mlsserver.service.impl.PredicateSetAssist;

import java.io.*;
import org.apache.hadoop.fs.FileSystem;
import java.util.*;

public class ParallelRuleDiscoveryTopkDiversified implements Serializable {
    private static final long serialVersionUID = -7355788782601134464L;
    private static Logger logger = LoggerFactory.getLogger(ParallelRuleDiscoveryTopkDiversified.class);

    // each predicate attaches corresponding data in ParsedColumn
    private List<Predicate> allPredicates;
    private ArrayList<Predicate> applicationRHSs;
    private int maxTupleNum; // the maximum number of tuples in a rule
    private long support;
    private float confidence;
    private long maxOneRelationNum;
    private long allCount;
    private InputLight inputLight;

    public int MIN_NUM_WORK_UNITS = 200000;

    // max number of partition of Lattice
    public static int NUM_LATTICE = 200;
    public int MAX_CURRENT_PREDICTES = 5; // the maximum |X| size in rule

    public static int MIN_LATTICE_PARALLEL = 1000;

    // generate all predicate set including different tuple ID pair
    private static final PredicateProviderIndex predicateProviderIndex = PredicateProviderIndex.getInstance();
    private int allPredicateSize;

    private HashMap<String, Long> tupleNumberRelations;

    // invalid predicates with supp < thr
    private HashSet<Predicate> invalidPredicates;

    // for top-k diversified
    private int K; // top-k num
    private int iteration;
    private Relevance relevance;
    private Diversity diversity;
    private double lambda;
    private double sampleRatioForTupleCov;
    private ArrayList<REE> final_results;  // final solution
    private double score_max;
    private REE ree_next;
    private CarryInfo info_next = null; // in current iteration, the information carried by ree_next
    private RelevanceModel relevance_model;

    private double w_supp;
    private double w_conf;
    private double w_rel_model;
    private double w_attr_non;
    private double w_pred_non;
    private double w_attr_dis;
    private double w_tuple_cov;

    // for top-k
    private boolean if_top_k = false; // whether diversity is empty
    private transient PriorityQueue<REE> topKREEsTemp;
    private Double[] topKREEScores;

    /*
    *  for top-k diversified.
    *  Note that, Q_past + intermediateREEs_past = the Q_past in the paper;  Q_past_current + intermediateREEs_current = the Q_past_current in the paper.
    */
    
    // Global for each iteration. Q_past saves all valid rules that have been expanded. Note that, Q_past + intermediateREEs = the real Q_past in draft.
    private HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> Q_past; // The form: <X size, X, p0, map>. The keys of map: "UB_rel", "rel_score", "UB", "UB_lazy", "LB_lazy". Note that, UB* and LB* for valid REEs is the real score, since valid rules will not be expanded further.
    // Local in each iteration. Q_past_current saves {Q_past + newly added valid rules} in current iteration.
    private HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> Q_past_current; // The form: <X size, X, p0, map>. The keys of map: "UB_rel", "rel_score", "UB", "UB_lazy", "LB_lazy". Note that, UB* and LB* for valid REEs is the real score, since valid rules will not be expanded further.

    // Global for each iteration. intermediateREEs_past saves all intermediate rules that have been expanded.
    private HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> intermediateREEs_past; // The form: <X size, X, p0, map>. The keys of map: "UB_rel", "rel_score", "UB", "UB_lazy", "LB_lazy"
    // Local in each iteration.  intermediateREEs_current saves the intermediate rules that are computed in current itetation.
    private HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> intermediateREEs_current; // The form: <X size, X, p0, map>. The keys of map: "UB_rel", "rel_score", "UB", "UB_lazy", "LB_lazy"

    // Local in each iteration. The REEs that should be expanded in the next iteration.  They will be pruned in lattice of this iteration, due to UB < score_max. In the next iteration, Q_next will be the initial lattice.
    private HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> Q_next;  // The form: <X size, X, RHS, map>

    private double[] queue_LB;
    private double ith_LB;  // Note that, the rule with ith-LB is needed, i.e., not pruned.

    // Global for each iteration.
//    private HashMap<Integer, HashSet<IBitSet>> invalidX; // the predicate set with support < threshold. keys: 0, 1, 2  -> t0, t1, t0t1
    private HashSet<IBitSet> invalidX;
    private HashMap<IBitSet, HashSet<Predicate>> invalidXRHSs;

    // Global for each iteration. The valid REEs with supp >= thr and conf >= thr. The relevance score is fixed, only diversity score may need to be re-computed. They will be pruned in lattice.
    private HashSet<REE> validREEs;
    private HashMap<IBitSet, HashSet<Predicate>> validREEsMap;  // <X, RHSs>
    // Note that, validLatticeVertexMap is for pruning, guided by Lemma 3.2 in paper TANE!
    private HashMap<Integer, HashMap<IBitSet, HashSet<Predicate>>> validLatticeVertexMap;   //  The form: <X size, <X, RHSs>>

    private Map<PredicateSet, List<Predicate>> validConstantRule = new HashMap<>();

    private boolean ifPrune; // 1: use pruning strategies; 0: not use

    private int if_cluster_workunits; // Not efficient! Just set it to 0!
    private int filter_enum_number;

    private int index_null_string;
    private int index_null_double;
    private int index_null_long;


    // return the final solution
    public ArrayList<REE> getFinalResults() {
        return this.final_results;
    }


    public ParallelRuleDiscoveryTopkDiversified(List<Predicate> predicates, int maxTupleNum, long support,
                                                float confidence, long maxOneRelationNum, Input input, long allCount,
                                                boolean ifPrune, int if_cluster_workunits,
                                                int filter_enum_number, int K,
                                                String relevance, String diversity, double lambda, double sampleRatioForTupleCov,
                                                double w_supp, double w_conf, double w_rel_model,
                                                double w_attr_non, double w_pred_non, double w_attr_dis, double w_tuple_cov,
                                                String predicateToIDFile, String relevanceModelFile, FileSystem hdfs,
                                                int index_null_string, int index_null_double, int index_null_long,
                                                int MAX_X_LENGTH) {
        this.allPredicates = predicates;
        this.maxTupleNum = maxTupleNum;
        this.support = support;
        this.confidence = confidence;
        this.maxOneRelationNum = maxOneRelationNum;
        this.allCount = allCount;
        this.inputLight = new InputLight(input);

        this.index_null_string = index_null_string;
        this.index_null_double = index_null_double;
        this.index_null_long = index_null_long;
        logger.info("index_null_string: {}", index_null_string);
        logger.info("index_null_double: {}", index_null_double);
        logger.info("index_null_long: {}", index_null_long);

        this.MAX_CURRENT_PREDICTES = MAX_X_LENGTH;
        logger.info("MAX_X_LENGTH: {}", MAX_CURRENT_PREDICTES);

        this.K = K;
        this.relevance = new Relevance(relevance);
        this.diversity = new Diversity(diversity);
        if (diversity.equals("null")) { // top-k
            this.if_top_k = true;
            this.topKREEsTemp = new PriorityQueue<REE>(new Comparator<REE>() {
                @Override
                public int compare(REE o1, REE o2) {
                    if (o1.getScore() < o2.getScore()) {
                        return 1;
                    } else if (o1.getScore() > o2.getScore()) {
                        return -1;
                    } else {
                        return 0;
                    }
                }
            });
            this.topKREEScores = new Double[this.K];
        }
        this.lambda = lambda;
        this.sampleRatioForTupleCov = sampleRatioForTupleCov;
        this.w_supp = w_supp;
        this.w_conf = w_conf;
        this.w_rel_model = w_rel_model;
        this.w_attr_non = w_attr_non;
        this.w_pred_non = w_pred_non;
        this.w_attr_dis = w_attr_dis;
        this.w_tuple_cov = w_tuple_cov;

        this.final_results = new ArrayList<>();

        this.ifPrune = ifPrune;

        this.if_cluster_workunits = if_cluster_workunits;

        this.filter_enum_number = filter_enum_number;

        // set support for each predicate;
        HashMap<String, HashMap<Integer, Long>> statistic = new HashMap<>();
        for (Predicate p : this.allPredicates) {
            // statistic
            // op1
            ParsedColumn<?> col1 = p.getOperand1().getColumn();
            if (!statistic.containsKey(col1.toStringData())) {
                HashMap<Integer, Long> temp = new HashMap<>();
                for (int i = 0; i < col1.getValueIntSize(); i++) {
                    if (!temp.containsKey(col1.getValueInt(i))) {
                        temp.put(col1.getValueInt(i), 1L);
                    } else {
                        temp.put(col1.getValueInt(i), temp.get(col1.getValueInt(i)) + 1);
                    }
                }
                statistic.put(col1.toStringData(), temp);
            }

            // op2
            ParsedColumn<?> col2 = p.getOperand2().getColumn();
            if (!statistic.containsKey(col2.toStringData())) {
                HashMap<Integer, Long> temp = new HashMap<>();
                for (int i = 0; i < col2.getValueIntSize(); i++) {
                    if (!temp.containsKey(col2.getValueInt(i))) {
                        temp.put(col2.getValueInt(i), 1L);
                    } else {
                        temp.put(col2.getValueInt(i), temp.get(col2.getValueInt(i)) + 1);
                    }
                }
                statistic.put(col2.toStringData(), temp);
            }
            // statistic support for predicate
            p.setSupportPredicate(statistic);
        }

        // invalid predicate combinations
        this.invalidX = new HashSet<>();
        this.invalidXRHSs = new HashMap<>();

        this.Q_past = new HashMap<>();
        this.Q_past_current = new HashMap<>();
        this.intermediateREEs_past = new HashMap<>();
        this.intermediateREEs_current = new HashMap<>();
        this.Q_next = new HashMap<>();

        this.validREEs = new HashSet<>();
        this.validREEsMap = new HashMap<>();
        this.validLatticeVertexMap = new HashMap<>();

        this.queue_LB = new double[this.K];
        for (int i = 0; i < this.K; i++) {
            this.queue_LB[i] = Double.NEGATIVE_INFINITY;
        }

        // record the invalid predicates with supp < thr
        this.invalidPredicates = new HashSet<>();
        for (Predicate p : this.allPredicates) {
            if (p.isConstant()) {
                if (p.getSupport() * this.maxOneRelationNum < this.support) {  // larger than the real support when in rule
//                    this.invalidX.add(new PredicateSet(p).getBitset());  // removing invalid predicates from allPredicates is enough.
                    this.invalidPredicates.add(p);
                }
            } else {
                if (p.getSupport() < this.support) {
//                    this.invalidX.add(new PredicateSet(p).getBitset());
                    this.invalidPredicates.add(p);
                }
            }
//            if (p.getTableName().contains("tax") && p.isConstant()) { // for tax, constants in constant predicate for singleexemp, marriedexemp, childexemp are meaningless
//                this.invalidPredicates.add(p);
//            }
        }
        logger.info("#### invalid Predicates num {}:  {}", this.invalidPredicates.size(), this.invalidPredicates);

        // remove the predicate with supp < thr
        for (Predicate p : this.invalidPredicates) {
            this.allPredicates.remove(p);
        }

        this.tupleNumberRelations = new HashMap<>();
        for (int i = 0; i < this.inputLight.getNames().size(); i++) {
            this.tupleNumberRelations.put(this.inputLight.getNames().get(i), (long) this.inputLight.getLineCounts()[i]);
        }

        // add to Predicate Set
        for (Predicate p : this.allPredicates) {
            predicateProviderIndex.addPredicate(p);
        }

        // load relevance model
        if (this.relevance.getCandidate_functions().contains("relevance_model")) {
            this.relevance_model = new RelevanceModel(predicateToIDFile, relevanceModelFile, hdfs);
        }
    }


    private ArrayList<Predicate> applicationDrivenSelection(List<Predicate> all_predicates) {
        int whole_num_nonCons = 0;
        int whole_num_cons = 0;
        List<Predicate> predicates = new ArrayList<>();

        for (Predicate rhs : all_predicates) {
            // when maxTupleNum <= 2, we do not consider t1 constant RHS, since it is redundant due to the symmetry!
            if (this.maxTupleNum <= 2 && rhs.isConstant() && rhs.getIndex1() == 1) {
                continue;
            }
//            if (rhs.isConstant() && rhs.getTableName().contains("tax")) {
//                continue;
//            }
            predicates.add(rhs);

            if (rhs.isConstant()) {
                whole_num_cons++;
            } else {
                whole_num_nonCons++;
            }
        }

        ArrayList<Predicate> applicationRHSs = new ArrayList<>();
        // 1 - choose the first few non-constant rhss
//        int NUM_rhs = 4;
//        if (NUM_rhs > whole_num_nonCons) {
//            NUM_rhs = whole_num_nonCons;
//        }
//        logger.info("#### choose the first {} rhss", NUM_rhs);
//        int count = 0;
//        HashMap<String, Predicate> temp = new HashMap<>();
//        for (Predicate p : predicates) {
//            if (p.isConstant()) {
//                continue;
//            }
//            if (!temp.containsKey(p.toString())) {
//                temp.put(p.toString(), p);
//            }
//        }
//
//        for (Map.Entry<String, Predicate> entry : temp.entrySet()) {
//            applicationRHSs.add(entry.getValue());
//            count++;
//            if (count == NUM_rhs) {
//                break;
//            }
//        }


        // 2 - choose some rhss with minimum support value
//        int NUM_rhs = 4;
//        if (NUM_rhs > predicates.size()) {
//            NUM_rhs = predicates.size();
//        }
//        logger.info("#### choose {} RHSs with minimum support value", NUM_rhs);
//        ArrayList<Long> allSupports = new ArrayList<>();
//        for (Predicate p : predicates) {
//            allSupports.add(p.getSupport());
//        }
//        allSupports.sort(Comparator.naturalOrder());
//        Long minSupp = allSupports.get(NUM_rhs - 1);
//        for (Predicate p : predicates) {
//            if (p.getSupport() <= minSupp) {
//                applicationRHSs.add(p);
//            }
//        }
//        if (applicationRHSs.size() > NUM_rhs) { // maybe exist redundant supp value minSupp
//            applicationRHSs.subList(0, NUM_rhs);
//        }


        // 3 - choose 4 rhss: one min-supp, one max-supp, and two random rhss
//        logger.info("#### choose 4 RHSs: one min-supp, one max-supp, and two random RHSs");
//        Collections.sort(predicates, new Comparator<Predicate>() {
//            @Override
//            public int compare(Predicate o1, Predicate o2) {
//                return o1.toString().compareTo(o2.toString());
//            }
//        });
//        Random rand = new Random();
//        rand.setSeed(1234567);
//        HashSet<Integer> random_idx = new HashSet<>();
//        while (random_idx.size() < 2) {
//            int idx = rand.nextInt(predicates.size());
//            random_idx.add(idx);
//        }
//        for (int choose_idx : random_idx) {
//            applicationRHSs.add(predicates.get(choose_idx));
//        }
//        ArrayList<Long> allSupports = new ArrayList<>();
//        for (Predicate p : predicates) {
//            allSupports.add(p.getSupport());
//        }
//        allSupports.sort(Comparator.naturalOrder());
//        Long minSupp = allSupports.get(0);
//        Long maxSupp = allSupports.get(allSupports.size() - 1);
//        if (applicationRHSs.get(0).getSupport() == minSupp || applicationRHSs.get(1).getSupport() == minSupp) {
//            minSupp = allSupports.get(1);
//        }
//        if (applicationRHSs.get(0).getSupport() == maxSupp || applicationRHSs.get(1).getSupport() == maxSupp) {
//            maxSupp = allSupports.get(allSupports.size() - 2);
//        }
//        for (Predicate p : predicates) {
//            if (applicationRHSs.contains(p)) {
//                continue;
//            }
//            if (p.getSupport() == minSupp || p.getSupport() == maxSupp) {
//                applicationRHSs.add(p);
//            }
//            if (applicationRHSs.size() == 4) {
//                break;
//            }
//        }

        // 4. randomly choose some RHS.
//        int NUM_rhs = 4;
//        if (NUM_rhs > predicates.size()) {
//            NUM_rhs = predicates.size();
//        }
//        logger.info("#### randomly choose {} RHSs", NUM_rhs);
//        Collections.sort(predicates, new Comparator<Predicate>() {
//            @Override
//            public int compare(Predicate o1, Predicate o2) {
//                return o1.toString().compareTo(o2.toString());
//            }
//        });
//        Random rand = new Random();
//        rand.setSeed(1234567);
//        HashSet<Integer> random_idx = new HashSet<>();
//        while (random_idx.size() < NUM_rhs) {
//            int idx = rand.nextInt(predicates.size());
//            random_idx.add(idx);
//        }
//        for (int choose_idx : random_idx) {
//            applicationRHSs.add(predicates.get(choose_idx));
//        }

        // 5. randomly choose (n+m) RHSs, including n nonConstant and m constant RHSs
//        int NUM_Constant = 3;
//        int NUM_NonConstant = 2;
//        if (NUM_Constant > whole_num_cons) {
//            NUM_Constant = whole_num_cons;
//        }
//        if (NUM_NonConstant > whole_num_nonCons) {
//            NUM_NonConstant = whole_num_nonCons;
//        }
//        logger.info("#### randomly choose {} RHSs, including {} nonConstant RHSs and {} constant RHSs", NUM_Constant + NUM_NonConstant, NUM_NonConstant, NUM_Constant);
//        Collections.sort(predicates, new Comparator<Predicate>() {
//            @Override
//            public int compare(Predicate o1, Predicate o2) {
//                return o1.toString().compareTo(o2.toString());
//            }
//        });
//        Random rand = new Random();
//        rand.setSeed(1234567);
//        HashSet<Integer> random_idx = new HashSet<>();
//        int constant_num = 0;
//        int nonConstant_num = 0;
//        while (random_idx.size() < (NUM_Constant + NUM_NonConstant)) {
//            int idx = rand.nextInt(predicates.size());
//            if (predicates.get(idx).isConstant()) {
//                if (constant_num < NUM_Constant && !random_idx.contains(idx)) {
//                    random_idx.add(idx);
//                    constant_num = constant_num + 1;
//                }
//            } else {
//                if (nonConstant_num < NUM_NonConstant && !random_idx.contains(idx)) {
//                    random_idx.add(idx);
//                    nonConstant_num = nonConstant_num + 1;
//                }
//            }
//        }
//        for (int choose_idx : random_idx) {
//            applicationRHSs.add(predicates.get(choose_idx));
//        }

        // 6. randomly choose some non-constant RHS.
//        int NUM_NonConstant = 4;
//        if (NUM_NonConstant > whole_num_nonCons) {
//            NUM_NonConstant = whole_num_nonCons;
//        }
//        logger.info("#### randomly choose {} non-constant RHSs", NUM_NonConstant);
//        Collections.sort(predicates, new Comparator<Predicate>() {
//            @Override
//            public int compare(Predicate o1, Predicate o2) {
//                return o1.toString().compareTo(o2.toString());
//            }
//        });
//        Random rand = new Random();
//        rand.setSeed(1234567);
//        HashSet<Integer> random_idx = new HashSet<>();
//        while (random_idx.size() < NUM_NonConstant) {
//            int idx = rand.nextInt(predicates.size());
//            if (!predicates.get(idx).isConstant()) {
//                random_idx.add(idx);
//            }
//        }
//        for (int choose_idx : random_idx) {
//            applicationRHSs.add(predicates.get(choose_idx));
//        }

        // 7. use all non-constant predicates as RHSs
//        logger.info("#### choose all non-constant RHSs");
//        for (Predicate p : predicates) {
//            if (!p.isConstant()) {
//                applicationRHSs.add(p);
//            }
//        }

        // 8. use all predicates as RHSs
        logger.info("#### choose all RHSs");
        for (Predicate p : predicates) {
//            if (!p.isConstant() && p.getOperand1().getColumnLight().getName().contains("scheduled_service")) {
//                continue;
//            }
//            if (p.isConstant() && p.getOperand1().getColumnLight().getName().contains("Inspection_Type") && p.getConstant().contains("Canvass")) {
//                applicationRHSs.add(p);
//                break;
//            }
            // for case study
//            if (p.getOperand1().getColumnLight().getName().equals("Emergency_Service")) {
//                applicationRHSs.add(p);
//            }
            applicationRHSs.add(p);
        }

        // 9. test NCVoter
//        logger.info("#### choose voting_intention as RHS");
//        for (Predicate p : predicates) {
//            if (p.getOperand1().toString(0).equals("t0.voting_intention")) {
//                applicationRHSs.add(p);
//            }
//        }

        // 10. test airports - fix RHSs
//        for (Predicate p : predicates) {
//            if (p.isConstant()) {
//                continue;
//            }
//            if (p.getOperand1().getColumnLight().getName().contains("wikipedia_link") ||
//                p.getOperand1().getColumnLight().getName().contains("home_link") ||
//                p.getOperand1().getColumnLight().getName().contains("iso_region") ||
//                p.getOperand1().getColumnLight().getName().contains("type")) {
//                applicationRHSs.add(p);
//            }
//        }

        // 11. test inspection - fix RHSs
//        for (Predicate p : predicates) {
//            if (p.isConstant()) {
//                continue;
//            }
//            if (p.getOperand1().getColumnLight().getName().contains("City") ||
//                p.getOperand1().getColumnLight().getName().contains("Inspection_ID") ||
//                p.getOperand1().getColumnLight().getName().contains("DBA_Name") ||
//                p.getOperand1().getColumnLight().getName().contains("Facility_Type") ||
//                p.getOperand1().getColumnLight().getName().contains("Latitude") ||
//                p.getOperand1().getColumnLight().getName().contains("Longitude") ||
//                p.getOperand1().getColumnLight().getName().contains("Results")) {
//                applicationRHSs.add(p);
//            }
//        }

        // 12. test ncvoter - fix RHSs
//        for (Predicate p : predicates) {
//            if (p.isConstant()) {
//                continue;
//            }
//            if (p.getOperand1().getColumnLight().getName().equals("city") ||
//                p.getOperand1().getColumnLight().getName().equals("city_id2") ||
//                p.getOperand1().getColumnLight().getName().equals("county_id") ||
//                p.getOperand1().getColumnLight().getName().equals("id")) {
//                applicationRHSs.add(p);
//            }
//        }

        logger.info("applicationRHSs size : {}", applicationRHSs.size());
        for (Predicate p : applicationRHSs) {
            logger.info("applicationRHSs: {}", p.toString());
        }
        return applicationRHSs;
    }

    /**
     * if RHS only contain non-constant predicates, then filter the irrelevant predicates
     * */
    private void filterIrrelevantPredicates(ArrayList<Predicate> applicationRHSs, List<Predicate> allPredicates) {
        // filter the predicates that are irrelevant to rhs
        boolean allNonConstant = true;
        HashSet<String> relationRHS = new HashSet<>();
        for (Predicate p : applicationRHSs) {
            if (p.isConstant()) {
                allNonConstant = false;
                break;
            }
            relationRHS.add(p.getOperand1().getColumn().getTableName());
            relationRHS.add(p.getOperand2().getColumn().getTableName());
        }
//        for (String name : relationRHS) {
//            logger.info("Table name: {}", name);
//        }
        if (allNonConstant) {
            ArrayList<Predicate> removePredicates = new ArrayList<>();
            for (Predicate p : allPredicates) {
                String name_1 = p.getOperand1().getColumn().getTableName();
                String name_2 = p.getOperand2().getColumn().getTableName();
//                logger.info("currPredicate: {}, name_1: {}, name_2: {}", p, name_1, name_2);
                if (!relationRHS.contains(name_1) || !relationRHS.contains(name_2)) {
                    removePredicates.add(p);
                }
            }
            logger.info("#### Filter Irrelevant Predicates size: {}", removePredicates.size());
            for (Predicate p : removePredicates) {
//                logger.info("remove predicate: {}", p);
                allPredicates.remove(p);
            }
        }
    }

    // remove the constant predicates related to Property_Feature dataset
    private void removePropertyFeatureCPredicates(List<Predicate> allPredicates) {
        ArrayList<Predicate> removePredicates = new ArrayList<>();
        for (Predicate p : allPredicates) {
            if (p.isConstant() && p.getOperand1().getColumn().getTableName().contains("Property_Features")) {
                removePredicates.add(p);
            }
        }
        logger.info("#### Filter Enum Constant Predicates size of Table Property_Features for X: {}", removePredicates.size());
        for (Predicate p : removePredicates) {
            allPredicates.remove(p);
        }
    }

    private void removeEnumPredicates(List<Predicate> allPredicates) {
        ArrayList<Predicate> removePredicates = new ArrayList<>();
        for (Predicate p : allPredicates) {
            if (!p.isConstant()) {
                continue;
            }
//            if (p.getOperand1().getColumnLight().getUniqueConstantNumber() <= this.filter_enum_number ||
//                p.getOperand2().getColumnLight().getUniqueConstantNumber() <= this.filter_enum_number) {
//                removePredicates.add(p);
//            }
            if (p.getOperand1().getColumnLight().getUniqueConstantNumberWithoutOnlyOneExistence() <= this.filter_enum_number ||
                    p.getOperand2().getColumnLight().getUniqueConstantNumberWithoutOnlyOneExistence() <= this.filter_enum_number) {
                removePredicates.add(p);
            }
        }
        logger.info("#### Filter Enum Constant Predicates size: {}", removePredicates.size());
        for (Predicate p : removePredicates) {
            allPredicates.remove(p);
        }
    }

    private void removeNonEnumPredicates(List<Predicate> allPredicates) {
        ArrayList<Predicate> removePredicates = new ArrayList<>();
        for (Predicate p : allPredicates) {
            // remove predicates with attribute such as id
            if (p.getOperand1().getColumnLight().getUniqueConstantNumber() > this.allCount * 0.8 ||
                    p.getOperand2().getColumnLight().getUniqueConstantNumber() > this.allCount * 0.8) {
                removePredicates.add(p);
            }
        }
        logger.info("#### Filter non-enum Constant Predicates size: {}", removePredicates.size());
        for (Predicate p : removePredicates) {
            allPredicates.remove(p);
        }
    }

    /**
     * merge work units by X
     */
    private ArrayList<WorkUnit> mergeWorkUnits(ArrayList<WorkUnit> workUnits) {
        ArrayList<WorkUnit> workUnits_res = new ArrayList<>();
        HashMap<PredicateSet, PredicateSet> dups = new HashMap<>();
        for (WorkUnit wu : workUnits) {
            if (dups.containsKey(wu.getCurrrent())) {
                dups.get(wu.getCurrrent()).or(wu.getRHSs());
            } else {
                PredicateSet ps = new PredicateSet(wu.getRHSs());
                dups.put(wu.getCurrrent(), ps);
            }
        }
        for (Map.Entry<PredicateSet, PredicateSet> entry : dups.entrySet()) {
            WorkUnit workUnit = new WorkUnit();
            for (Predicate p : entry.getKey()) {
                workUnit.addCurrent(p);
            }
            for (Predicate p : entry.getValue()) {
                workUnit.addRHS(p);
            }
            workUnits_res.add(workUnit);
        }
        return workUnits_res;
    }

    public boolean levelwiseRuleDiscoveryFirstIteration(String taskId, SparkSession spark, int iteration) {

        this.iteration = iteration;
        this.score_max = Double.NEGATIVE_INFINITY;
        this.ree_next = null;

        // default support ratio: 10^-6, confidence: 0.75
        if (!this.ifPrune) { // for no pruning strategies baseline, large MIN_NUM_WORK_UNITS cause GC Time OUT; Otherwise, for our method, MIN_NUM_WORK_UNITS=20w
            this.MIN_NUM_WORK_UNITS = 3000;
            if (taskId.contains("inspection")) {
                if (this.confidence > 0.9) {
                    this.MIN_NUM_WORK_UNITS = 1000;
                } else if (this.confidence > 0.75) {
                    this.MIN_NUM_WORK_UNITS = 2000;
                }
            }
            else if (taskId.contains("ncvoter")) {
                this.MIN_NUM_WORK_UNITS = 800;
//                // vary k
//                if (this.K > 10) {
//                    this.MIN_NUM_WORK_UNITS = 1000;
//                } else {
//                    // vary conf
//                    if (this.confidence > 0.75) {
//                        this.MIN_NUM_WORK_UNITS = 1000;
//                        if (this.confidence > 0.9) {
//                            this.MIN_NUM_WORK_UNITS = 500;
//                        }
//                    } else {
//                        // vary supp
//                        this.MIN_NUM_WORK_UNITS = 1000;
//                    }
//                }
            }
            else if (taskId.contains("tax")) {
                this.MIN_NUM_WORK_UNITS = 1000;
            }
            else if (taskId.contains("aminer")) {
                this.MIN_NUM_WORK_UNITS = 1000;
            }
        }
//        else { // top-k-div
//            if (taskId.contains("ncvoter")) {
//                if (this.confidence > 0.85) {
//                    this.MIN_NUM_WORK_UNITS = 1000;
//                }
//            }
//        }
        logger.info("#### MIN_NUM_WORK_UNITS: {}", this.MIN_NUM_WORK_UNITS);

//        if (table_name.contains("Property_Features")) {
//            removePropertyFeatureCPredicates(this.allPredicates);
//        }

        this.prepareAllPredicatesMultiTuples();

        // remove "t.paper_affiliations = ." for dblp data
        ArrayList<Predicate> rm_preds = new ArrayList<>();
        for (Predicate p : this.allPredicates) {
            if (p.isConstant() && p.getTableName().contains("AMiner") &&
                    p.getOperand1().getColumnLight().getName().contains("paper_affiliations") && p.getConstant().contains(".")) {
                rm_preds.add(p);
            }
//            if (p.getTableName().contains("tax") && p.getOperand1().getColumnLight().getName().contains("zip")) { // supp < 10^-4
//                rm_preds.add(p);
//            }
//            if (p.getTableName().contains("tax") && p.getOperand1().getColumnLight().getName().contains("phone")) { // supp < 10^-4
//                rm_preds.add(p);
//            }
        }
        for (Predicate p : rm_preds) {
            this.allPredicates.remove(p);
        }

        this.applicationRHSs = this.applicationDrivenSelection(this.allPredicates);

        this.removeNonEnumPredicates(this.allPredicates);

        this.removeEnumPredicates(this.allPredicates);

        // remove predicates that are irrelevant to RHSs
        if (this.maxTupleNum <= 2) {
            this.filterIrrelevantPredicates(this.applicationRHSs, this.allPredicates);
        }

        logger.info("Parallel Mining with Predicates size {} and Predicates {}", this.allPredicates.size(), this.allPredicates);
        int cpsize = 0;
        int psize = 0;
        for (Predicate p : this.allPredicates) {
            if (p.isConstant()) {
                cpsize++;
            } else {
                psize++;
            }
        }
        logger.info("#### after filtering, there are {} predicates, constant size: {}, non-constant size: {}", this.allPredicates.size(), cpsize, psize);


        // initialize the 1st level combinations
        Lattice lattice = new Lattice(this.maxTupleNum);
        lattice.initialize(this.allPredicates, this.maxTupleNum, this.applicationRHSs);

        String option = "original";
        if (!this.ifPrune) {
            option = "none";
        }

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        PredicateSetAssist psAssist = new PredicateSetAssist();
        psAssist.setIndexProvider(PredicateSet.indexProvider);//在这里设置，这个值肯定是确定了的
        psAssist.setBf(PredicateSet.bf);
        psAssist.setTaskId(taskId);
        Broadcast<PredicateSetAssist> bcpsAssist = sc.broadcast(psAssist);

        int level = 0;
        while (lattice != null && level <= MAX_CURRENT_PREDICTES) {  // level == |X|
            logger.info("#### search level : {}, score_max: {}", level, this.score_max);

            ArrayList<WorkUnit> workUnits_init = new ArrayList<>();
            long start = System.currentTimeMillis();
            // when level = 0, the result returned by lattice.generateWorkUnits() is empty.
            if (level != 0) {
                // collect a set of work units
                workUnits_init = lattice.generateWorkUnits();

                // merge work units with the same X
                workUnits_init = this.mergeWorkUnits(workUnits_init);

                // prune work units from Q_next, i.e., that are intermediate rules, due to UB_lazy <= this.score_max
                if (this.ifPrune) {
                    workUnits_init = this.pruneWorkUnits(workUnits_init);
                }
            }
            logger.info("#### generate work units, size: {}, using time: {}", workUnits_init.size(), (System.currentTimeMillis() - start) / 1000);

            HashMap<PredicateSet, Double> currentSupports = new HashMap<>();

            if (taskId.contains("tax")) {
//                if (level < 2) {
//                    if (taskId.contains("tax200")) {
//                        this.MIN_NUM_WORK_UNITS = workUnits_init.size() * 8;
//                    } else if (taskId.contains("tax400")) {
//                        this.MIN_NUM_WORK_UNITS = 2000;
//                    } else if (taskId.contains("tax600")) {
//                        this.MIN_NUM_WORK_UNITS = 1400;
//                    } else if (taskId.contains("tax800")) {
//                        this.MIN_NUM_WORK_UNITS = 20000;
//                    } else if (taskId.contains("tax1000")) {
//                        this.MIN_NUM_WORK_UNITS = 100000;
//                    }
//                } else {
//                    if (taskId.contains("tax400")) {
//                        this.MIN_NUM_WORK_UNITS = 2000;
//                    }
//                    this.MIN_NUM_WORK_UNITS = 200000;
//                }
//                if (!this.ifPrune) {
//                    if (level < 2) {
//                        this.MIN_NUM_WORK_UNITS = ((workUnits_init.size() + 1) / 10);
//                    } else if (level == 2) {
//                        this.MIN_NUM_WORK_UNITS = ((workUnits_init.size() + 1) / 8);
//                    } else {
//                        this.MIN_NUM_WORK_UNITS = 1000;
//                    }
//                } else {
//                    if (level <= 2) {
//                        this.MIN_NUM_WORK_UNITS = ((workUnits_init.size() + 1) / 2);
//                    } else {
//                        this.MIN_NUM_WORK_UNITS = 6000;
//                    }
//                }
                logger.info("level:{} , MIN_NUM_WORK_UNITS:{}", level, this.MIN_NUM_WORK_UNITS);
            }
//            else if (taskId.contains("ncvoter")) {
//                if (this.ifPrune) {
//                    if (this.confidence > 0.85 && this.confidence < 0.95) { // top-k-div, vary conf, conf=0.9
//                        if (level == 4) {
//                            this.MIN_NUM_WORK_UNITS = 300;
//                        }
//                        if (level == 5) {
//                            this.MIN_NUM_WORK_UNITS = 450;
//                        }
//                    }
//                }
//                logger.info("level:{} , MIN_NUM_WORK_UNITS:{}", level, this.MIN_NUM_WORK_UNITS);
//            }

            while (workUnits_init.size() != 0) {
                // solve skewness
//                start = System.currentTimeMillis();
                ArrayList<WorkUnit> workUnits = new ArrayList<>();
                int remainStartSc = this.solveSkewness(workUnits_init, workUnits);  // copy work units, with different combinations of pid
//                logger.info("#### solveSkeness time: {}", (System.currentTimeMillis() - start) / 1000);

                // validate all work units
                start = System.currentTimeMillis();
                List<Message> messages = null;
                if (workUnits.size() > 0) {
                    messages = this.run(workUnits, taskId, sc, bcpsAssist);
                    logger.info("Integrate messages ...");
                    messages = this.integrateMessages_new(messages);
                }
                logger.info("#### run and integrateMessages time: {}", (System.currentTimeMillis() - start) / 1000);

                // maintain top-1 REE with maximum score
//                start = System.currentTimeMillis();
                if (messages != null) {
                    for (Message message : messages) {
                        // compute score and UB of work units, maintain the rule with maximum score
                        this.computeScoresAndUB(message);

                        // update validConstantRule
                        for (Predicate p : message.getCurrent()) {
                            if (p.isConstant()) {
                                for (Predicate rhs : message.getValidRHSs()) {
                                    this.validConstantRule.putIfAbsent(message.getCurrentSet(), new ArrayList<>());
                                    this.validConstantRule.get(message.getCurrentSet()).add(rhs);
                                }
                                break;
                            }
                        }
                    }
                }
//                logger.info("#### computeScoresAndUB time: {}", (System.currentTimeMillis() - start) / 1000);

                // collect invalid X
                this.addInvalidX(messages);

                if (option.equals("original")) {
                    if (remainStartSc < workUnits_init.size()) {
                        // collect supports of candidate rules
                        this.getCandidateXSupportRatios(currentSupports, messages);
                        // prune meaningless work units
                        workUnits_init = this.pruneXWorkUnits(currentSupports, workUnits_init, remainStartSc);
                    } else {
                        workUnits_init = new ArrayList<>();
                    }
                } else {
                    // obtain the left work units
                    ArrayList<WorkUnit> remainTasks = new ArrayList<>();
                    for (int tid = remainStartSc; tid < workUnits_init.size(); tid++) {
                        remainTasks.add(workUnits_init.get(tid));
                    }
                    workUnits_init = remainTasks;
                }
            }

            if (level + 1 > MAX_CURRENT_PREDICTES) {
                this.intermediateREEs_current.clear();
                break;
            }

            if (this.ifPrune) {
//                start = System.currentTimeMillis();
                // (1) update Q_next, add ones with UB score < score_max; (2) update intermediateREEs_past
                this.updateQnextAndIntermediateREEsPast();

                // clear intermediateREEs_current
                this.intermediateREEs_current.clear();

//                logger.info("### update prunedREEsByUB time: {}", (System.currentTimeMillis() - start) / 1000);
            }

            start = System.currentTimeMillis();
            Lattice nextLattice = runNextLattices(lattice, spark);
            logger.info("#### runNextLattices time: {}", (System.currentTimeMillis() - start) / 1000);

            if (nextLattice == null || nextLattice.size() == 0) {
                break;
            }
            lattice = nextLattice;
            level++;
        }

        if (this.ree_next == null) {
            return false;
        }

        logger.info("#### the next REE discovered: {}, supp: {}, conf: {}, rel_score: {}, div_score: {}, score: {}",
                this.ree_next.toString(), this.ree_next.getSupport(), this.ree_next.getConfidence(),
                this.ree_next.getRelevance_score(), this.ree_next.getDiversity_score(), this.ree_next.getScore());

        // update the diversity information covered by the current partial solution
//        long start = System.currentTimeMillis();
        this.diversity.updateCurrentPartialSolutionDiv(this.ree_next, this.info_next, this.final_results);
//        logger.info("### updateCurrentPartialSolutionDiv time: {}", (System.currentTimeMillis() - start) / 1000);

        // print diversity information of the current partial solution
        this.diversity.printInfo();

        // add next ree
        this.final_results.add(this.ree_next);

        // remove next ree from Q_past_current and validREEs
        PredicateSet ps = this.ree_next.getCurrentList();
        this.Q_past_current.get(ps.size()).get(ps.getBitset()).remove(this.ree_next.getRHS());
        this.validREEs.remove(this.ree_next);

        // update Q_past
        this.deepCopy(this.Q_past_current, this.Q_past);

        // update LB, w.r.t. changed Sigma, when diversity measures contain attribute_distance
        if (this.ifPrune && this.diversity.getCandidate_functions().contains("attribute_distance")) {
//            start = System.currentTimeMillis();
            this.updateLBValues();
//            logger.info("### when diversity contains attribute distance, update LB time: {}", (System.currentTimeMillis() - start) / 1000);
        }

        return true;

    }


    // copy map1 -> map2
    public void deepCopy(HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> map, HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> result) {
        result.clear();
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : map.entrySet()) {
            int X_size = entry1.getKey();
            result.putIfAbsent(X_size, new HashMap<>());
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet X_set = entry2.getKey();
                result.get(X_size).putIfAbsent(X_set, new HashMap<>());
                for (Map.Entry<Predicate, HashMap<String, Double>> entry : entry2.getValue().entrySet()) {
                    Predicate rhs = entry.getKey();
                    result.get(X_size).get(X_set).put(rhs, entry.getValue());
                }
            }
        }
    }


    public void printInfos(HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> info) {
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : info.entrySet()) {
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet X_set = entry2.getKey();
                for (Map.Entry<Predicate, HashMap<String, Double>> entry : entry2.getValue().entrySet()) {
                    Predicate rhs = entry.getKey();
                    logger.info("#### X: {},\tY:{},\tinfo: {}", new PredicateSet(X_set), rhs, entry.getValue());
                }
            }
        }
    }


    public boolean levelwiseRuleDiscovery(String taskId, SparkSession spark, int iteration) {

        this.iteration = iteration;
        this.score_max = Double.NEGATIVE_INFINITY;
        this.ree_next = null;

        this.validConstantRule.clear();  // At the beginning of each iteration, clear validConstantRule!!!

//        if (this.ifPrune && taskId.contains("ncvoter")) {
//            if (this.confidence > 0.85 && this.confidence < 0.95) { // top-k-div, vary conf, conf=0.9
//                this.MIN_NUM_WORK_UNITS = 200000;
//            }
//            logger.info("MIN_NUM_WORK_UNITS:{}", this.MIN_NUM_WORK_UNITS);
//        }
        if (taskId.contains("tax")) {
            if (this.ifPrune) {
                this.MIN_NUM_WORK_UNITS = 200000;
            } else {
                this.MIN_NUM_WORK_UNITS = 2000;
            }
        }

        long start = System.currentTimeMillis();
        if (this.ifPrune) {
            // update the LB queue and i-th LB value
            this.updateLBQueue();
            this.ith_LB = this.queue_LB[this.K - this.iteration - 1];
            logger.info("#### ith_LB: {}", this.ith_LB);

            // prune Q_past, Q_next and intermediate_REEs_past by ith_LB
            this.pruneByLBLazy();
        }
        logger.info("### LB time: {}", (System.currentTimeMillis() - start) / 1000);

        // copy Q_past as Q_past_current
        this.deepCopy(this.Q_past, this.Q_past_current);
//        logger.info("#### Q_past:");
//        printInfos(this.Q_past);


        String option = "original";
        if (!this.ifPrune) {
            option = "none";
        }

//        start = System.currentTimeMillis();
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        PredicateSetAssist psAssist = new PredicateSetAssist();
        psAssist.setIndexProvider(PredicateSet.indexProvider);//在这里设置，这个值肯定是确定了的
        psAssist.setBf(PredicateSet.bf);
        psAssist.setTaskId(taskId);
        Broadcast<PredicateSetAssist> bcpsAssist = sc.broadcast(psAssist);
//        logger.info("#### broadcast prepare, using time: {}", (System.currentTimeMillis() - start) / 1000);


        int level = 0;

        // generate lattice from Q_next, and clear Q_next!
        HashMap<Integer, Lattice> lattice_initial = this.generateLattice();
        Lattice lattice = lattice_initial.get(level);  // Here, lattice is null
        this.Q_next.clear();

//        logger.info("----- initial lattice:");
//        for (LatticeVertex lv : lattice.getLatticeLevel().values()) {
//            logger.info("#### {}", lv.printCurrent());
//        }

        // Here, level=0 means re-compute score in Q_past; Otherwise, level == the length of lv (LHS+RHS) in lattice
        while (level <= MAX_CURRENT_PREDICTES + 1) {  // level == |{X, p0}| == |X| + 1
            logger.info("#### search level : {}, score_max: {}", level, this.score_max);

            if (this.Q_past.isEmpty() && (lattice == null || lattice.size() == 0)) { // Q_past.isEmpty() means level>0
                level++;

                lattice = lattice_initial.get(level);
                if (lattice != null) {
                    // pruned by valid, invalid, Q_next.   NO NEED!
//                    lattice.pruneLatticeNew(this.invalidX, this.invalidXRHSs, this.Q_next,
//                            this.validREEsMap, this.validLatticeVertexMap, this.if_UB_prune);
//                    lattice.pruneInvalidLatticeNew(this.invalidX);

                    // pruned by lazy evaluation and update Q_next, since score_max changes
                    if (this.ifPrune) {
                        lattice.pruneLatticeBeforeComputeScore(this.intermediateREEs_past, this.Q_next, this.lambda, this.score_max,
                                this.allPredicates, this.allPredicateSize);
                    }
                }

                continue;
            }

            ArrayList<WorkUnit> workUnits_init = new ArrayList<>();

//            start = System.currentTimeMillis();
            if (this.Q_past.isEmpty()) {

//                logger.info("----- current lattice:");
//                for (LatticeVertex lv : lattice.getLatticeLevel().values()) {
//                    logger.info("#### {}", lv.printCurrent());
//                }

                // (2) After update score_max by re-computing scores of valid REEs. collect a set of work units from Q_next, i.e., lattice
                workUnits_init = lattice.generateWorkUnits();

                // merge work units with the same X
                workUnits_init = this.mergeWorkUnits(workUnits_init);

                // prune work units from Q_next, i.e., that are intermediate rules, due to UB_lazy <= this.score_max
                if (this.ifPrune) {
                    workUnits_init = this.pruneWorkUnits(workUnits_init);
                }
//                logger.info("#### generate, merge and prune work units from Q/lattice, size: {}, using time: {}", workUnits_init.size(), (System.currentTimeMillis() - start) / 1000);
            }
            else {
                // (1) First re-compute scores of valid REEs!!! collect work units from Q_past, and clear Q_past!  This step will only be conducted once.
//                start = System.currentTimeMillis();
                workUnits_init = this.generateAndPruneWorkUnits();
//                logger.info("#### collect work units from Q_past, work units size: {}, using time: {}", workUnits_Qpast.size(), (System.currentTimeMillis() - start) / 1000);

                // clear Q_past, since all work units from Q_past will be computed below
                this.Q_past.clear();

                if (this.diversity.getCandidate_functions().contains("tuple_coverage")) {
//                    start = System.currentTimeMillis();
                    workUnits_init = this.mergeWorkUnits(workUnits_init);
//                    logger.info("#### merge work units, size:{}, time: {}", workUnits_init.size(), (System.currentTimeMillis() - start) / 1000);
                }
            }

//            logger.info("#### ------------ work units: ");
//            for (WorkUnit wu : workUnits_init) {
//                logger.info("X: {}\tY: {}", wu.getCurrrent(), wu.getRHSs());
//            }

            logger.info("#### work units size: {}", workUnits_init.size());
            HashMap<PredicateSet, Double> currentSupports = new HashMap<>();
            while (!workUnits_init.isEmpty()) {
                // solve skewness
//                start = System.currentTimeMillis();
                ArrayList<WorkUnit> workUnits = new ArrayList<>();
                int remainStartSc = this.solveSkewness(workUnits_init, workUnits);  // copy work units, with different combinations of pid
//                logger.info("#### solveSkeness time: {}", (System.currentTimeMillis() - start) / 1000);

                // validate all work units
                start = System.currentTimeMillis();
                List<Message> messages = null;
                if (workUnits.size() > 0) {
                    messages = this.run(workUnits, taskId, sc, bcpsAssist);
                    logger.info("Integrate messages ...");
                    messages = this.integrateMessages_new(messages);
                }
                logger.info("#### run and integrateMessages time: {}", (System.currentTimeMillis() - start) / 1000);

                // maintain top-1 REE with maximum score
//                start = System.currentTimeMillis();
                if (messages != null) {
                    for (Message message : messages) {
                        // compute score and UB of work units, maintain the rule with maximum score
                        this.computeScoresAndUB(message);

                        // update validConstantRule
                        for (Predicate p : message.getCurrent()) {
                            if (p.isConstant()) {
                                for (Predicate rhs : message.getValidRHSs()) {
                                    this.validConstantRule.putIfAbsent(message.getCurrentSet(), new ArrayList<>());
                                    this.validConstantRule.get(message.getCurrentSet()).add(rhs);
                                }
                                break;
                            }
                        }
                    }
                }
//                logger.info("#### computeScoresAndUB time: {}", (System.currentTimeMillis() - start) / 1000);

                // collect invalid X
                this.addInvalidX(messages);

                if (option.equals("original")) {
                    if (remainStartSc < workUnits_init.size()) {
                        // collect supports of candidate rules
                        this.getCandidateXSupportRatios(currentSupports, messages);
                        // prune meaningless work units
                        workUnits_init = this.pruneXWorkUnits(currentSupports, workUnits_init, remainStartSc);
                    } else {
                        workUnits_init = new ArrayList<>();
                    }
                } else {
                    // obtain the left work units
                    ArrayList<WorkUnit> remainTasks = new ArrayList<>();
                    for (int tid = remainStartSc; tid < workUnits_init.size(); tid++) {
                        remainTasks.add(workUnits_init.get(tid));
                    }
                    workUnits_init = remainTasks;
                }
            }

            if (level == 0) { // when level = 0, we only re-compute valid rules. The work units of lattice have not been dealt with!!!
                level++;
                continue;
            }

            if (level == MAX_CURRENT_PREDICTES + 1) {
                this.intermediateREEs_current.clear();
                break;
            }

            if (this.ifPrune) {
//                start = System.currentTimeMillis();
                // (1) update Q_next, add ones with UB score < score_max; (2) update intermediateREEs_past(has done in computeScoresAndUB()!)
                this.updateQnextAndIntermediateREEsPast();

                // clear intermediateREEs_current
                this.intermediateREEs_current.clear();

//                logger.info("### update prunedREEsByUB time: {}", (System.currentTimeMillis() - start) / 1000);
            }

            start = System.currentTimeMillis();
            Lattice nextLattice = runNextLattices(lattice, spark);
            logger.info("#### runNextLattices time: {}", (System.currentTimeMillis() - start) / 1000);

            level++;

            Lattice lattice_last_iter = lattice_initial.get(level);
            if (lattice_last_iter != null) {

                // prune by valid, invalid, Q_next.   NO NEED!
//                lattice_last_iter.pruneLatticeNew(this.invalidX, this.invalidXRHSs, this.Q_next,
//                        this.validREEsMap, this.validLatticeVertexMap, this.ifPrune);
//                lattice_last_iter.pruneInvalidLatticeNew(this.invalidX);

                // prune by lazy evaluation and update Q_next, since score_max changes
                if (this.ifPrune) {
                    lattice_last_iter.pruneLatticeBeforeComputeScore(this.intermediateREEs_past, this.Q_next, this.lambda, this.score_max,
                            this.allPredicates, this.allPredicateSize);
                }

                // merge lattice
                lattice_last_iter.addLatticeNew(nextLattice);

                lattice = lattice_last_iter;
            }
            else {
                lattice = nextLattice;
            }

        }

        if (this.ree_next == null) {
            if (!this.validREEs.isEmpty()) {
                logger.info("[Error] there is no valid REE newly discovered, while validREEs is not empty, which is wrong!");
                System.exit(0);
            }
            logger.info("[Note] there is no valid REE newly discovered, so we terminate the iteration!");
            return false;
        }

        logger.info("#### the next REE discovered: {}, supp: {}, conf: {}, rel_score: {}, div_score: {}, score: {}",
                this.ree_next.toString(), this.ree_next.getSupport(), this.ree_next.getConfidence(),
                this.ree_next.getRelevance_score(), this.ree_next.getDiversity_score(), this.ree_next.getScore());

        // update the diversity information covered by the current partial solution
//        start = System.currentTimeMillis();
        this.diversity.updateCurrentPartialSolutionDiv(this.ree_next, this.info_next, this.final_results);
//        logger.info("### updateCurrentPartialSolutionDiv time: {}", (System.currentTimeMillis() - start) / 1000);

        // print diversity information of the current partial solution
        this.diversity.printInfo();

        // add next ree
        this.final_results.add(this.ree_next);

        // remove next ree from Q_past_current and validREEs
        PredicateSet ps = this.ree_next.getCurrentList();
        this.Q_past_current.get(ps.size()).get(ps.getBitset()).remove(this.ree_next.getRHS());
        this.validREEs.remove(this.ree_next);

        // copy Q_past_current to Q_past
        this.deepCopy(this.Q_past_current, this.Q_past);

        // update LB, w.r.t. changed Sigma, when diversity measures contain attribute_distance
        if (this.ifPrune && this.diversity.getCandidate_functions().contains("attribute_distance")) {
//            start = System.currentTimeMillis();
            this.updateLBValues();
//            logger.info("### when diversity contains attribute distance, update LB time: {}", (System.currentTimeMillis() - start) / 1000);
        }

        return true;

    }


    public void levelwiseRuleDiscoveryTopK(String taskId, SparkSession spark, int iteration) {

        this.iteration = iteration;
//        if (table_name.contains("Property_Features")) {
//            removePropertyFeatureCPredicates(this.allPredicates);
//        }

        this.prepareAllPredicatesMultiTuples();

        // remove "t.paper_affiliations = ." for dblp data
        ArrayList<Predicate> rm_preds = new ArrayList<>();
        for (Predicate p : this.allPredicates) {
            if (p.isConstant() && p.getTableName().contains("AMiner") &&
                    p.getOperand1().getColumnLight().getName().contains("paper_affiliations") && p.getConstant().contains(".")) {
                rm_preds.add(p);
            }
//            if (p.getTableName().contains("tax") && p.getOperand1().getColumnLight().getName().contains("zip")) { // supp < 10^-4
//                rm_preds.add(p);
//            }
//            if (p.getTableName().contains("tax") && p.getOperand1().getColumnLight().getName().contains("phone")) { // supp < 10^-4
//                rm_preds.add(p);
//            }
        }
        for (Predicate p : rm_preds) {
            this.allPredicates.remove(p);
        }

        this.applicationRHSs = this.applicationDrivenSelection(this.allPredicates);

        this.removeNonEnumPredicates(this.allPredicates);

        this.removeEnumPredicates(this.allPredicates);

        // remove predicates that are irrelevant to RHSs
        if (this.maxTupleNum <= 2) {
            this.filterIrrelevantPredicates(this.applicationRHSs, this.allPredicates);
        }

        logger.info("Parallel Mining with Predicates size {} and Predicates {}", this.allPredicates.size(), this.allPredicates);
        int cpsize = 0;
        int psize = 0;
        for (Predicate p : this.allPredicates) {
            if (p.isConstant()) {
                cpsize++;
            } else {
                psize++;
            }
        }
        logger.info("#### after filtering, there are {} predicates, constant size: {}, non-constant size: {}", this.allPredicates.size(), cpsize, psize);


        // initialize the 1st level combinations
        Lattice lattice = new Lattice(this.maxTupleNum);
        lattice.initialize(this.allPredicates, this.maxTupleNum, this.applicationRHSs);

        String option = "original";
        if (!this.ifPrune) {
            option = "none";
        }

        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        PredicateSetAssist psAssist = new PredicateSetAssist();
        psAssist.setIndexProvider(PredicateSet.indexProvider);//在这里设置，这个值肯定是确定了的
        psAssist.setBf(PredicateSet.bf);
        psAssist.setTaskId(taskId);
        Broadcast<PredicateSetAssist> bcpsAssist = sc.broadcast(psAssist);

        int level = 0;
        while (lattice != null && level <= MAX_CURRENT_PREDICTES) {
            logger.info("Search level : {}", level);

            ArrayList<WorkUnit> workUnits_init = new ArrayList<>();
            // when level = 0, the result returned by lattice.generateWorkUnits() is empty.
            long start = System.currentTimeMillis();
            if (level != 0) {
                // collect a set of work units
                workUnits_init = lattice.generateWorkUnits();
                workUnits_init = this.mergeWorkUnits(workUnits_init);
            }
            long end = System.currentTimeMillis();
            logger.info("#### generate work units, size: {}, using time: {}", workUnits_init.size(), (end - start) / 1000);

            // batch mode
            HashMap<PredicateSet, Double> currentSupports = new HashMap<>();

            boolean ifTerm = true;
            while (workUnits_init.size() != 0) {
                // solve skewness
                ArrayList<WorkUnit> workUnits = new ArrayList<>();
                int remainStartSc = this.solveSkewness(workUnits_init, workUnits);  // copy work units, with different combinations of pid

                // validate all work units
                start = System.currentTimeMillis();
                List<Message> messages = null;
                if (workUnits.size() > 0) {
                    messages = this.run(workUnits, taskId, sc, bcpsAssist);
                    logger.info("Integrate messages ...");
                    messages = this.integrateMessages_new(messages);
                    ifTerm = false;
                }
                logger.info("#### run and integrateMessages time: {}", (System.currentTimeMillis() - start) / 1000);

                // maintain top-1 REE with maximum score
                if (messages != null) {
                    for (Message message : messages) {
                        // compute score and UB of work units, maintain the rule with maximum score
                        this.maintainTopKRules(message);

                        // update validConstantRule
                        for (Predicate p : message.getCurrent()) {
                            if (p.isConstant()) {
                                for (Predicate rhs : message.getValidRHSs()) {
                                    this.validConstantRule.putIfAbsent(message.getCurrentSet(), new ArrayList<>());
                                    this.validConstantRule.get(message.getCurrentSet()).add(rhs);
//                                    logger.info("Valid REE values ###### are : {}, supp : {}, conf : {}", ree.toString(), ree.getSupport(), ree.getConfidence());
                                }
                                break;
                            }
                        }
                    }
                }
                // collect invalid X
                this.addInvalidX(messages);
                if (option.equals("original")) {
                    if (remainStartSc < workUnits_init.size()) {
                        // collect supports of candidate rules
                        this.getCandidateXSupportRatios(currentSupports, messages);
                        // prune meaningless work units
                        workUnits_init = this.pruneXWorkUnits(currentSupports, workUnits_init, remainStartSc);
                    } else {
                        workUnits_init = new ArrayList<>();
                    }
                } else {
                    // obtain the left work units
                    ArrayList<WorkUnit> remainTasks = new ArrayList<>();
                    for (int tid = remainStartSc; tid < workUnits_init.size(); tid++) {
                        remainTasks.add(workUnits_init.get(tid));
                    }
                    workUnits_init = remainTasks;
                }

            }

            if (level > 0 && ifTerm) {
                break;
            }

            if (level + 1 > MAX_CURRENT_PREDICTES) {
                break;
            }

            start = System.currentTimeMillis();
            Lattice nextLattice = runNextLattices(lattice, spark);
            logger.info("#### runNextLattices time: {}", (System.currentTimeMillis() - start) / 1000);

            if (nextLattice == null || nextLattice.size() == 0) {
                break;
            }
            lattice = nextLattice;
            level++;
        }

        for (int i = 0; i < this.K; i++) {
            if (this.topKREEsTemp.size() == 0) {
                break;
            }
            this.final_results.add(this.topKREEsTemp.poll());
        }

    }


    public HashMap<Integer, Lattice> generateLattice() {
//        Lattice lattice = new Lattice(this.maxTupleNum);
        HashMap<Integer, Lattice> lattice_initial = new HashMap<>();
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.Q_next.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();

                    PredicateSet temp = new PredicateSet(Xset);
                    temp.add(rhs);
                    LatticeVertex lv = new LatticeVertex(temp, this.maxTupleNum); // Notice the max_predicate_index_in_X is NOT used when iteration > 0!
                    lv.addRHS(rhs);

                    for (Predicate p : temp) {
                        if (p.isConstant()) {
                            continue;
                        }
                        lv.updateLatestTupleIDs(p.getIndex1(), p.getIndex2());
                        String r_1 = p.getOperand1().getColumn().getTableName();
                        String r_2 = p.getOperand2().getColumn().getTableName();
                        lv.updateTIDRelationName(r_1, r_2);
                    }

//                    lattice.addLatticeVertexNew(lv);  // add RHSs, not and RHSs!!!
                    lattice_initial.putIfAbsent(temp.size(), new Lattice(this.maxTupleNum));
                    lattice_initial.get(temp.size()).addLatticeVertexNew(lv);
                }
            }
        }

//        return lattice;
        return lattice_initial;
    }

    // generate work units from Q_past, and prune ones with UB_lazy <= this.score_max
    public ArrayList<WorkUnit> generateAndPruneWorkUnits() {
        ArrayList<WorkUnit> workUnits = new ArrayList<>();
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.Q_past.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();
                    // prune work units with UB_lazy <= score_max
                    if (entry_.getValue().get("UB_lazy") <= this.score_max) {  // no use, since we first re-compute the scores of valid rules at level 0!
                        continue;
                    }
                    // generate work units
                    boolean compute_div_only = !this.diversity.getCandidate_functions().contains("tuple_coverage");
                    WorkUnit workUnit = new WorkUnit(new PredicateSet(Xset), rhs, compute_div_only);
                    workUnits.add(workUnit);
                }
            }
        }

        return workUnits;
    }

    // Since Sigma changed, i.e., added a new REE, we update all LB value of valid REEs when diversity measures contain attribute_distance
    public void updateLBValues() {
        HashMap<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> Q_past_temp = new HashMap<>();
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.Q_past.entrySet()) {
            int Xset_size = entry1.getKey();

            Q_past_temp.putIfAbsent(Xset_size, new HashMap<>());
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();

                Q_past_temp.get(Xset_size).putIfAbsent(Xset, new HashMap<>());
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();
                    HashMap<String, Double> info_ = entry_.getValue();

                    double rel_score = info_.get("rel_score");
                    double LB_div_new = this.diversity.computeLB(new PredicateSet(Xset), rhs, MAX_CURRENT_PREDICTES, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
                    double LB_lazy_new = rel_score + this.lambda * LB_div_new;

                    info_.put("LB_lazy", LB_lazy_new);
                    Q_past_temp.get(Xset_size).get(Xset).put(rhs, info_);
                }
            }
        }
        this.Q_past = Q_past_temp;
    }

    public void updateLBQueue() {
        // get all LB value of valid REEs
        PriorityQueue<Double> queue_LB_temp = new PriorityQueue<>(Comparator.reverseOrder());
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.Q_past.entrySet()) {
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                for (Map.Entry<Predicate, HashMap<String, Double>> entry : entry2.getValue().entrySet()) {
                    double LB_lazy = entry.getValue().get("LB_lazy");
                    queue_LB_temp.add(LB_lazy);
                }
            }
        }

        // reset this.queue_LB
        for (int i = 0; i < this.K; i++) {
            this.queue_LB[i] = Double.NEGATIVE_INFINITY;
        }
        // update this.queue_LB
        for (int i = 0; i < this.K; i++) {
            if (queue_LB_temp.size() == 0) {
                break;
            }
            this.queue_LB[i] = queue_LB_temp.poll();
        }
    }

    // prune REEs with UB <= ith_LB
    public void pruneByLBLazy() {
        int count = 0;
        // 1. prune Q_past (valid REEs)
        HashMap<Integer, HashMap<IBitSet, HashSet<Predicate>>> removeREEs = new HashMap<>();
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.Q_past.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();
                    HashMap<String, Double> info_ = entry_.getValue();
                    if (info_.get("UB_lazy") < this.ith_LB) {
                        removeREEs.putIfAbsent(Xset_size, new HashMap<>());
                        removeREEs.get(Xset_size).putIfAbsent(Xset, new HashSet<>());
                        removeREEs.get(Xset_size).get(Xset).add(rhs);
                    }
                }
            }
        }
        for (Map.Entry<Integer, HashMap<IBitSet, HashSet<Predicate>>> entry1 : removeREEs.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashSet<Predicate>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Predicate rhs : entry2.getValue()) {
                    this.Q_past.get(Xset_size).get(Xset).remove(rhs);
//                    logger.info("#### Q_past remove X:{}, Y:{}", new PredicateSet(Xset), rhs);
                    count++;
                }
                if (this.Q_past.get(Xset_size).get(Xset).isEmpty()) {
                    this.Q_past.get(Xset_size).remove(Xset);
                }
            }
            if (this.Q_past.get(Xset_size).isEmpty()) {
                this.Q_past.remove(Xset_size);
            }
        }
        logger.info("#### Q_past remove size: {}, remaining key size: {}", count, this.Q_past.size());

        // 2. prune Q_next (intermediate REEs)
        count = 0;
        removeREEs.clear();
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.Q_next.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();
                    HashMap<String, Double> info_ = entry_.getValue();
                    if (info_ == null) {
                        continue;
                    }
                    if ((info_.get("UB_rel") + this.lambda * info_.get("UB_div_lazy")) < this.ith_LB) {
                        removeREEs.putIfAbsent(Xset_size, new HashMap<>());
                        removeREEs.get(Xset_size).putIfAbsent(Xset, new HashSet<>());
                        removeREEs.get(Xset_size).get(Xset).add(rhs);
                    }
                }
            }
        }
        for (Map.Entry<Integer, HashMap<IBitSet, HashSet<Predicate>>> entry1 : removeREEs.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashSet<Predicate>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Predicate rhs : entry2.getValue()) {
                    this.Q_next.get(Xset_size).get(Xset).remove(rhs);
//                    logger.info("#### Q_next remove X:{}, Y:{}", new PredicateSet(Xset), rhs);
                    count++;
                }
                if (this.Q_next.get(Xset_size).get(Xset).isEmpty()) {
                    this.Q_next.get(Xset_size).remove(Xset);
                }
            }
            if (this.Q_next.get(Xset_size).isEmpty()) {
                this.Q_next.remove(Xset_size);
            }
        }
        logger.info("#### Q_next remove size: {}, remaining key size: {}", count, this.Q_next.size());

        // 3. prune intermediate_REEs_past
        removeREEs.clear();
        count = 0;
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.intermediateREEs_past.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();
                    HashMap<String, Double> info_ = entry_.getValue();
                    if ((info_.get("UB_rel") + this.lambda * info_.get("UB_div_lazy")) < this.ith_LB) {
                        removeREEs.putIfAbsent(Xset_size, new HashMap<>());
                        removeREEs.get(Xset_size).putIfAbsent(Xset, new HashSet<>());
                        removeREEs.get(Xset_size).get(Xset).add(rhs);
                    }
                }
            }
        }
        for (Map.Entry<Integer, HashMap<IBitSet, HashSet<Predicate>>> entry1 : removeREEs.entrySet()) {
            int Xset_size = entry1.getKey();
            for (Map.Entry<IBitSet, HashSet<Predicate>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
                for (Predicate rhs : entry2.getValue()) {
                    this.intermediateREEs_past.get(Xset_size).get(Xset).remove(rhs);
//                    logger.info("#### intermediateREEs_past remove X:{}, Y:{}", new PredicateSet(Xset), rhs);
                    count++;
                }
                if (this.intermediateREEs_past.get(Xset_size).get(Xset).isEmpty()) {
                    this.intermediateREEs_past.get(Xset_size).remove(Xset);
                }
            }
            if (this.intermediateREEs_past.get(Xset_size).isEmpty()) {
                this.intermediateREEs_past.remove(Xset_size);
            }
        }
        logger.info("#### intermediateREEs_past remove size: {}, remaining key size: {}", count, this.intermediateREEs_past.size());

    }

    public void getNextRuleFromValidREEs() {
        logger.info("#### get next rule from the valid REEs set.");
        for (REE ree : this.validREEs) {
            // relevance score is fixed, not changed as the size of partial solution increase.
            double rel_score = ree.getRelevance_score();

            // compute the diversity score, i.e., the marginal diversity gain w.r.t. the current partial solution.
            double div_score = this.diversity.getDivScore(ree, this.final_results, MAX_CURRENT_PREDICTES, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);

            double score = rel_score + this.lambda * div_score;
            ree.setDiversityScore(div_score);
            ree.setScore(score);

            // update rule with maximum score
            if (score > this.score_max) {
                this.ree_next = ree;
                this.score_max = score;
            }
        }
    }

    private void computeScoresAndUB(Message message) {
        PredicateSet Xset = message.getCurrentSet();
        PredicateSet rhsList = message.getRHSList();
        long lhsSupport = message.getCurrentSupp();

        int Xset_size = Xset.size();
        IBitSet Xset_bs = Xset.getBitset();

        // update div_score and UB_div in Q_past_current
        if (message.isCompute_div_only()) { // diversity is not empty, compute diversity only and without tuple coverage
            for (Predicate rhs : rhsList) {

                boolean isValidREE = false;
                if (this.validREEsMap.containsKey(Xset_bs) && this.validREEsMap.get(Xset_bs).contains(rhs)) {
                    isValidREE = true;
                }

                // update Q_past_current
                HashMap<String, Double> info_;
                if (isValidREE) {
                    info_ = this.Q_past_current.get(Xset_size).get(Xset_bs).get(rhs);
                } else {
                    info_ = this.intermediateREEs_past.get(Xset_size).get(Xset_bs).get(rhs);
                }
                double rel_score = info_.get("rel_score");
                double UB_rel = info_.get("UB_rel");
//                double UB = info_.get("UB");
//                double UB_lazy = info_.get("UB_lazy");
//                double LB_lazy = info_.get("LB_lazy");
//                double UB_div_lazy = info_.get("UB_div_lazy");

                HashMap<String, Double> div_score_all = this.computeDivScore(message, rhs);
                double score_new = rel_score + this.lambda * div_score_all.get("div_score");

                double div_LB = this.diversity.computeLB(Xset, rhs, MAX_CURRENT_PREDICTES, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
                double LB_lazy_new = rel_score + this.lambda * div_LB;

                double UB_div = this.diversity.computeUB(div_score_all, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
                double UB_new = UB_rel + this.lambda * UB_div;
                double UB_lazy_new = rel_score + this.lambda * UB_div;

                HashMap<String, Double> info_new_ = new HashMap<>();
                info_new_.put("rel_score", rel_score);
                info_new_.put("UB_rel", UB_rel);
                info_new_.put("UB", UB_new);
                info_new_.put("UB_lazy", UB_lazy_new);
                info_new_.put("LB_lazy", LB_lazy_new);
                info_new_.put("UB_div_lazy", UB_div);
                if (isValidREE) {
                    // update Q_past_current
                    this.Q_past_current.get(Xset_size).get(Xset_bs).put(rhs, info_new_);
                } else {
                    // update intermediateREEs_past
                    this.intermediateREEs_past.get(Xset_size).get(Xset_bs).put(rhs, info_new_);
                    // add to intermediateREEs_current
                    this.intermediateREEs_current.putIfAbsent(Xset_size, new HashMap<>());
                    this.intermediateREEs_current.get(Xset_size).putIfAbsent(Xset_bs, new HashMap<>());
                    this.intermediateREEs_current.get(Xset_size).get(Xset_bs).put(rhs, info_new_);

//                    logger.info("#### intermediate REE-2: {} -> {}, info_new_: {}", Xset, rhs, info_new_);
                }

                if (!isValidREE) {
                    continue;
                }

                // update validREEs and score_max, ree_next
                for (REE ree : this.validREEs) {
                    if (!ree.getCurrentList().equals(Xset) || !ree.getRHS().equals(rhs)) {
                        continue;
                    }
                    ree.setAttr_nonoverlap_score(div_score_all.get("attribute_nonoverlap"));
                    ree.setPred_nonoverlap_score(div_score_all.get("predicate_nonoverlap"));
                    ree.setTuple_cov_score(div_score_all.get("tuple_coverage"));
                    ree.setAttr_distance_score(div_score_all.get("attribute_distance"));
                    ree.setDiversityScore(div_score_all.get("div_score"));
                    ree.setScore(score_new);  // real score
                    if (ree.getScore() > this.score_max) {
                        // for case study
//                        boolean isMeaningless = false;
//                        if (ree.getCurrentList().size() == 1) {
//                            for (Predicate p_X : ree.getCurrentList()) {
//                                if (!p_X.isConstant() && ree.getRHS().isConstant()) {
//                                    isMeaningless = true;
//                                }
//                            }
//                        }
//                        if (!isMeaningless) {
                            this.score_max = ree.getScore();
                            this.ree_next = ree;
                            this.info_next = message.getCarriedInfoLight(rhs);
//                        }
                    }

                    logger.info("#### valid REE-2: {}, supp: {}, conf: {}, rel_score:{}, div_score:{}, score: {}", ree.toString(), ree.getSupport(), ree.getConfidence(), ree.getRelevance_score(), ree.getDiversity_score(), ree.getScore());

                    if (this.diversity.getCandidate_functions().contains("attribute_distance")) {
                        // loose score, since attribute distance do not have anti-monotonicity!
                        double div_score_loose = div_score_all.get("div_score") - this.w_attr_dis * div_score_all.get("attribute_distance") + this.w_attr_dis * 0.0; // 0.0 is the attr_dis UB.
                        double score_new_loose = rel_score + this.lambda * div_score_loose;
                        HashMap<String, Double> info_revise_ = this.Q_past_current.get(Xset_size).get(Xset_bs).get(rhs);
                        info_revise_.put("UB", score_new_loose);
                        info_revise_.put("UB_lazy", score_new_loose);
                        info_revise_.put("UB_div_lazy", div_score_loose);
                        this.Q_past_current.get(Xset_size).get(Xset_bs).put(rhs, info_revise_);  // replace
                    }
                }
            }
            return;
        }

        // update intermediate REEs, i.e., supp satisfied, but conf not satisfied.
        for (Map.Entry<Integer, Long> entry : message.getIntermediateRHSsSupport().entrySet()) {
            Predicate rhs = PredicateSet.getPredicate(entry.getKey());
            long supp = entry.getValue();
            double conf = message.getIntermediateRHSsConfidence(entry.getKey());

            REE ree = this.transformREEs(message, Xset, rhs, supp, lhsSupport, conf); // intermediate ree, supp >= thr, conf < thr

            // compute UB and UB_lazy
            double UB_rel = this.relevance.computeUB(ree, this.w_supp, this.w_conf, this.relevance_model, this.w_rel_model);
            double UB_div = this.diversity.computeUB(ree, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
            double UB = UB_rel + this.lambda * UB_div;
            double UB_lazy = ree.getRelevance_score() + this.lambda * UB_div;

            // compute LB_lazy
            double LB_div = this.diversity.computeLB(ree, MAX_CURRENT_PREDICTES, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
            double LB_lazy = ree.getRelevance_score() + this.lambda * LB_div;

            HashMap<String, Double> info_ = new HashMap<>();
            info_.put("UB_rel", UB_rel);
            info_.put("rel_score", ree.getRelevance_score());
            info_.put("UB", UB);
            info_.put("UB_lazy", UB_lazy);
            info_.put("LB_lazy", LB_lazy);
            info_.put("UB_div_lazy", UB_div);

            this.intermediateREEs_current.putIfAbsent(Xset_size, new HashMap<>());
            this.intermediateREEs_current.get(Xset_size).putIfAbsent(Xset_bs, new HashMap<>());
            this.intermediateREEs_current.get(Xset_size).get(Xset_bs).put(rhs, info_);

            this.intermediateREEs_past.putIfAbsent(Xset_size, new HashMap<>());
            this.intermediateREEs_past.get(Xset_size).putIfAbsent(Xset_bs, new HashMap<>());
            this.intermediateREEs_past.get(Xset_size).get(Xset_bs).put(rhs, info_);

//            logger.info("#### intermediate REE-1: {}, supp:{}, conf: {}, info_: {}", ree.toString(), supp, conf, info_);
        }

        // update valid REEs in Q_past_current, i.e., supp satisfied, and conf satisfied.
        Predicate rhs_with_max_score = null;
        for (int i = 0; i < message.getValidRHSs().size(); i++) {
            Predicate rhs = message.getValidRHSs().get(i);
            long supp = message.getSupports().get(i);
            double conf = message.getConfidences().get(i);

            REE ree = this.transformREEs(message, Xset, rhs, supp, lhsSupport, conf);

            logger.info("#### valid REE-1: {}, supp: {}, conf: {}, rel_score:{}, div_score:{}, score: {}", ree.toString(), ree.getSupport(), ree.getConfidence(), ree.getRelevance_score(), ree.getDiversity_score(), ree.getScore());

            if (ree.getScore() > this.score_max) {
                // for case study
//                boolean isMeaningless = false;
//                if (ree.getCurrentList().size() == 1) {
//                    for (Predicate p_X : ree.getCurrentList()) {
//                        if (!p_X.isConstant() && ree.getRHS().isConstant()) {
//                            isMeaningless = true;
//                        }
//                    }
//                }
//                if (!isMeaningless) {
                    this.score_max = ree.getScore();
                    this.ree_next = ree;
                    rhs_with_max_score = rhs;
//                }
            }

            // compute UB. No need, since valid rules will not be expanded further.
//            double UB_rel = this.relevance.computeUB(ree, this.w_supp, this.w_conf, this.relevance_model, this.w_rel_model);
//            double UB_div = this.diversity.computeUB(ree, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
//            double UB = UB_rel + this.lambda * UB_div;
//            double UB_lazy = ree.getRelevance_score() + this.lambda * UB_div;

            double LB_div = this.diversity.computeLB(ree, MAX_CURRENT_PREDICTES, this.w_attr_non, this.w_pred_non, this.w_attr_dis, this.w_tuple_cov);
            double LB_lazy = ree.getRelevance_score() + this.lambda * LB_div;

            // update Q_past_current
            if (this.diversity.getCandidate_functions().contains("attribute_distance")) {
                // attribute distance do not have anti-monotonicity!
                int pred_id = PredicateSet.getIndex(rhs);
                double attr_distance_score = message.getAttribute_distance(pred_id);
                double div_UBscore = ree.getDiversity_score() - this.w_attr_dis * attr_distance_score + this.w_attr_dis * 0.0;  // 0.0 is the attr_dis UB.
                double score_new = ree.getRelevance_score() + this.lambda * div_UBscore;  // more loose
                this.Q_past_current.putIfAbsent(Xset_size, new HashMap<>());
                this.Q_past_current.get(Xset_size).putIfAbsent(Xset_bs, new HashMap<>());
                HashMap<String, Double> info_ = new HashMap<>();
//                info_.put("UB_rel", UB_rel);
                info_.put("UB_rel", ree.getRelevance_score());
                info_.put("rel_score", ree.getRelevance_score());
//                info_.put("UB", UB);
                info_.put("UB", score_new); // save real score instead of UB, since valid rules will not be expanded further.
//                info_.put("UB_lazy", UB_lazy);
                info_.put("UB_lazy", score_new); // save real score instead of UB, since valid rules will not be expanded further.
                info_.put("LB_lazy", LB_lazy);
                info_.put("UB_div_lazy", ree.getDiversity_score()); // save real score instead of UB, since valid rules will not be expanded further.
                this.Q_past_current.get(Xset_size).get(Xset_bs).put(rhs, info_);
            }
            else {
                this.Q_past_current.putIfAbsent(Xset_size, new HashMap<>());
                this.Q_past_current.get(Xset_size).putIfAbsent(Xset_bs, new HashMap<>());
                HashMap<String, Double> info_ = new HashMap<>();
//                info_.put("UB_rel", UB_rel);
                info_.put("UB_rel", ree.getRelevance_score());
                info_.put("rel_score", ree.getRelevance_score());
//                info_.put("UB", UB);
                info_.put("UB", ree.getScore()); // save real score instead of UB, since valid rules will not be expanded further.
//                info_.put("UB_lazy", UB_lazy);
                info_.put("UB_lazy", ree.getScore()); // save real score instead of UB, since valid rules will not be expanded further.
                info_.put("LB_lazy", LB_lazy);
                info_.put("UB_div_lazy", ree.getDiversity_score()); // save real score instead of UB, since valid rules will not be expanded further.
                this.Q_past_current.get(Xset_size).get(Xset_bs).put(rhs, info_);
            }

            this.validREEs.add(ree);

            this.validREEsMap.putIfAbsent(Xset_bs, new HashSet<>());
            this.validREEsMap.get(Xset_bs).add(rhs);

            if (conf == 1.0) {
                PredicateSet temp = new PredicateSet(Xset);
                temp.add(rhs);
                this.validLatticeVertexMap.putIfAbsent(temp.size(), new HashMap<>());
                this.validLatticeVertexMap.get(temp.size()).putIfAbsent(temp.getBitset(), new HashSet<>());
                this.validLatticeVertexMap.get(temp.size()).get(temp.getBitset()).add(rhs);
            }
        }
        if (rhs_with_max_score != null) {
            this.info_next = message.getCarriedInfoLight(rhs_with_max_score);
        }
    }

    public REE transformREEs(Message message, PredicateSet Xset, Predicate rhs,
                             long supp, long lhsSupport, double conf) {
        ArrayList<String> rel_funcs = this.relevance.getCandidate_functions();
        ArrayList<String> div_funcs = this.diversity.getCandidate_functions();
        double rel_score = 0.0;
        double div_score = 0.0;

        // compute relevance score
        double supp_score = 0;
        double rel_model_score = 0;
        for (String rel : rel_funcs) {
            switch (rel) {
                case "support":
                    supp_score = supp * 1.0 / this.allCount / this.allCount;
                    rel_score += this.w_supp * supp_score;
                    break;
                case "confidence":
                    rel_score += this.w_conf * conf;
                    break;
                case "relevance_model":
                    rel_model_score = this.relevance_model.predict_score(Xset, rhs);
                    rel_score += this.w_rel_model * rel_model_score;
                    break;
            }
        }

        // compute diversity score
        double attr_nonoverlap_score = 0;
        double pred_nonoverlap_score = 0;
        double tuple_cov_score = 0;
        double attr_distance_score = 0;
        for (String div : div_funcs) {
            int pred_id = PredicateSet.getIndex(rhs);
            switch (div) {
                case "attribute_nonoverlap":
                    attr_nonoverlap_score = message.getAttribute_nonoverlap(pred_id) * 1.0 / MAX_CURRENT_PREDICTES;
                    div_score += this.w_attr_non * attr_nonoverlap_score;
                    break;
                case "predicate_nonoverlap":
                    pred_nonoverlap_score = message.getPredicate_nonoverlap(pred_id) * 1.0 / MAX_CURRENT_PREDICTES;
                    div_score += this.w_pred_non * pred_nonoverlap_score;
                    break;
                case "tuple_coverage":
                    tuple_cov_score = message.getMarginalTupleCoverage(pred_id) * 1.0 / this.allCount / this.allCount;
                    div_score += this.w_tuple_cov * tuple_cov_score;
                    break;
                case "attribute_distance":
                    attr_distance_score = message.getAttribute_distance(pred_id);
                    div_score += this.w_attr_dis * attr_distance_score;
                    break;
            }
        }

        // update ree with maximum score
        double score = rel_score + this.lambda * div_score;
        REE ree = new REE(Xset, rhs, lhsSupport, supp, lhsSupport-supp, conf, rel_score, div_score, score,
                supp_score, attr_nonoverlap_score, pred_nonoverlap_score, tuple_cov_score, attr_distance_score);

        return ree;
    }

    public HashMap<String, Double> computeDivScore(Message message, Predicate rhs) {
        ArrayList<String> div_funcs = this.diversity.getCandidate_functions();
        double div_score = 0.0;

        HashMap<String, Double> div_score_all = new HashMap<>();
        // initialize
        div_score_all.put("attribute_nonoverlap", 0.0);
        div_score_all.put("predicate_nonoverlap", 0.0);
        div_score_all.put("tuple_coverage", 0.0);
        div_score_all.put("attribute_distance", 0.0);
        div_score_all.put("div_score", 0.0);

        // compute diversity score
        for (String div : div_funcs) {
            int pred_id = PredicateSet.getIndex(rhs);
            switch (div) {
                case "attribute_nonoverlap":
                    double attr_nonoverlap_score = message.getAttribute_nonoverlap(pred_id) * 1.0 / MAX_CURRENT_PREDICTES;
                    div_score_all.put("attribute_nonoverlap", attr_nonoverlap_score);
                    div_score += this.w_attr_non * attr_nonoverlap_score;
                    break;
                case "predicate_nonoverlap":
                    double pred_nonoverlap_score = message.getPredicate_nonoverlap(pred_id) * 1.0 / MAX_CURRENT_PREDICTES;
                    div_score_all.put("predicate_nonoverlap", pred_nonoverlap_score);
                    div_score += this.w_pred_non * pred_nonoverlap_score;
                    break;
                case "tuple_coverage":
                    double tuple_cov_score = message.getMarginalTupleCoverage(pred_id) * 1.0 / this.allCount / this.allCount;
                    div_score_all.put("tuple_coverage", tuple_cov_score);
                    div_score += this.w_tuple_cov * tuple_cov_score;
                    break;
                case "attribute_distance":
                    double attr_distance_score = message.getAttribute_distance(pred_id);
                    div_score_all.put("attribute_distance", attr_distance_score);
                    div_score += this.w_attr_dis * attr_distance_score;
                    break;
            }
        }
        div_score_all.put("div_score", div_score);
        return div_score_all;
    }

    private void maintainTopKRules(Message message) {
        PredicateSet Xset = message.getCurrentSet();
        PredicateSet rhsList = message.getRHSList();
        long lhsSupport = message.getCurrentSupp();

        for (int i = 0; i < message.getValidRHSs().size(); i++) {
            Predicate rhs = message.getValidRHSs().get(i);
            long supp = message.getSupports().get(i);
            double conf = message.getConfidences().get(i);

            REE ree = this.transformREEs(message, Xset, rhs, supp, lhsSupport, conf);

            logger.info("#### valid REE: {}, supp:{}, conf: {}, rel_score:{}, score: {}", ree.toString(), ree.getSupport(), ree.getConfidence(), ree.getRelevance_score(), ree.getScore());

            this.validREEsMap.putIfAbsent(Xset.getBitset(), new HashSet<>());
            this.validREEsMap.get(Xset.getBitset()).add(rhs);

            if (conf == 1.0) {
                PredicateSet temp = new PredicateSet(Xset);
                temp.add(rhs);
                this.validLatticeVertexMap.putIfAbsent(temp.size(), new HashMap<>());
                this.validLatticeVertexMap.get(temp.size()).putIfAbsent(temp.getBitset(), new HashSet<>());
                this.validLatticeVertexMap.get(temp.size()).get(temp.getBitset()).add(rhs);
            }

            this.topKREEsTemp.add(ree);
        }

        REE[] topKREEs = new REE[this.K];
        // score the current top-K REEs and their corresponding scores
        for (int i = 0; i < this.K; i++) {
            if (this.topKREEsTemp.size() == 0) {
                break;
            }
            topKREEs[i] = this.topKREEsTemp.poll();
            this.topKREEScores[i] = topKREEs[i].getScore();
        }
        // maintain the current top-K REEs in the temp max heap
        this.topKREEsTemp.clear();
        for (int i = 0; i < this.K; i++) {
            if (topKREEs[i] != null) {
                this.topKREEsTemp.add(topKREEs[i]);
            }
        }
    }

    /*
        prune work units using interestingness scores and supports
     */
    private ArrayList<WorkUnit> pruneXWorkUnits(Interestingness interestingness, double KthScore, HashMap<PredicateSet, Double> supp_ratios,
                                                ArrayList<WorkUnit> tasks, int remainedStartSc, String topKOption) {
        logger.info("#### all workunits num: {}, remainedStartSc: {}", tasks.size(), remainedStartSc);
        logger.info("#### before pruning, workunits num: {}", tasks.size() - remainedStartSc);
        ArrayList<WorkUnit> prunedTasks = new ArrayList<>();
        for (int tid = remainedStartSc; tid < tasks.size(); tid++) {
            WorkUnit task = tasks.get(tid);

            if (supp_ratios.containsKey(task.getCurrrent())) {
                // check support
                long suppOne = (long) (supp_ratios.get(task.getCurrrent()) * this.allCount);
                if (suppOne < support) {
                    continue;
                }
                // check interestingness UB
                for (Predicate rhs : task.getRHSs()) {
                    double ub = interestingness.computeUB(suppOne * 1.0, 1.0, task.getCurrrent(), rhs, topKOption);
                    if (ub <= KthScore) {
                        task.getRHSs().remove(rhs);
                    }
                }
                // if no RHSs to do in this task, continue to handle other ones
                if (task.getRHSs().size() == 0) {
                    continue;
                }
            }
            prunedTasks.add(task);
        }
        logger.info("#### after pruning, workunits num: {}", prunedTasks.size());
        return prunedTasks;
    }

    private ArrayList<WorkUnit> pruneXWorkUnits(HashMap<PredicateSet, Double> supp_ratios, ArrayList<WorkUnit> tasks, int remainedStartSc) {
        logger.info("#### all workunits num: {}, remainedStartSc: {}", tasks.size(), remainedStartSc);
        logger.info("#### before pruning, workunits num: {}", tasks.size() - remainedStartSc);
        ArrayList<WorkUnit> prunedTasks = new ArrayList<>();
        for (int tid = remainedStartSc; tid < tasks.size(); tid++) {
            WorkUnit task = tasks.get(tid);
            IBitSet Xset = task.getCurrrent().getBitset();

            // check support
            if (supp_ratios.containsKey(task.getCurrrent())) {
                long suppOne = (long) (supp_ratios.get(task.getCurrrent()) * this.allCount);
                if (suppOne < this.support) {
                    continue;
                }
                // check UB for top-k, which only run one iteration
//                if (this.if_top_k) {
//                    for (Predicate rhs : task.getRHSs()) {
//                        double ub = this.w_supp * suppOne * 1.0 / this.allCount / this.allCount + this.w_conf * 1.0;
//                        if (ub <= this.topKREEScores[this.K - 1]) {
//                            task.getRHSs().remove(rhs);
//                        }
//                    }
//                }
            }
            if (!this.if_top_k) {
                for (Predicate rhs : task.getRHSs()) {
                    // check invalid rhs
                    if (this.invalidXRHSs.containsKey(Xset) && this.invalidXRHSs.get(Xset).contains(rhs)) {
                        task.getRHSs().remove(rhs);
                    }
                    // check UB_lazy
                    if (this.pruneByUBLazy(Xset, rhs)) {
                        task.getRHSs().remove(rhs);
                    }
                }
            }
            // if no RHSs to do in this task, continue to handle other ones
            if (task.getRHSs().size() == 0) {
                continue;
            }
            prunedTasks.add(task);
        }
        logger.info("#### after pruning, workunits num: {}", prunedTasks.size());
        return prunedTasks;
    }

    // remove rhs due to UB_lazy <= this.score_max
    public boolean pruneByUBLazy(IBitSet Xset, Predicate rhs) {
        if (!this.ifPrune) {
            return false;
        }
        int Xset_size = new PredicateSet(Xset).size();
        if (this.validREEsMap.containsKey(Xset) && this.validREEsMap.get(Xset).contains(rhs)) {
            // valid
            HashMap<String, Double> info_ = this.Q_past_current.get(Xset_size).get(Xset).get(rhs);
            if (info_.get("UB_lazy") <= this.score_max) {
                return true;
            }
        } else {
            // intermediate
            if (this.intermediateREEs_past.containsKey(Xset_size) && this.intermediateREEs_past.get(Xset_size).containsKey(rhs)) {
                HashMap<String, Double> info_ = this.intermediateREEs_past.get(Xset_size).get(Xset).get(rhs);
                if (info_.get("UB_lazy") <= this.score_max) {
                    return true;
                }
            }
        }
        return false;
    }

    public Lattice runNextLattices(Lattice lattice, SparkSession spark) {

        if (lattice.size() == 0) {
            return null;
        }

        // prune invalid lattice nodes, using support and UB. Note that, in this step, we can use validREEsMap and validLatticeVertexMap to prune! Because valid rees will not be expanded further.
        long start_pruneLattice_time = System.currentTimeMillis();
        if (this.iteration == 0) {  // (1) top-k, or (2) the first iteration of top-k diversified
            lattice.pruneLattice(this.invalidX, this.invalidXRHSs, this.Q_next, this.validREEsMap, this.validLatticeVertexMap,
                    this.ifPrune, this.if_top_k);
        } else {
            lattice.pruneLatticeNew(this.invalidX, this.invalidXRHSs, this.Q_next, this.validREEsMap, this.validLatticeVertexMap,
                    this.ifPrune);
        }
        logger.info("#### pruneLattice time: {}", (System.currentTimeMillis() - start_pruneLattice_time) / 1000);

//        logger.info("----- after pruneLattice");
//        for (LatticeVertex lv : lattice.getLatticeLevel().values()) {
//            logger.info("#### {}", lv.printCurrent());
//        }

        if (lattice.size() == 0) {
            return null;
        }

        // if lattice.size() <= MIN_LATTICE_PARALLEL, we do not use spark to generate next lattice parallel
        if (lattice.size() <= MIN_LATTICE_PARALLEL) {
            long start = System.currentTimeMillis();
            Lattice nextLattice = lattice.generateNextLatticeLevel(this.allPredicates, this.invalidX,
                    this.invalidXRHSs, this.validREEsMap, this.validLatticeVertexMap,
                    predicateProviderIndex, this.iteration,
                    this.Q_next, this.ifPrune);

            // pruning by RHS intersection
            if (this.iteration == 0) {  // (1) top-k, or (2) the first iteration of top-k diversified
                nextLattice.removeInvalidLatticeAndRHSs(lattice);
            }
            else {
                // Note that, we do NOT use allLatticeVertexBits to prune, when iteration != 0
                if (this.ifPrune) {
                    nextLattice.pruneLatticeBeforeComputeScore(this.intermediateREEs_past, this.Q_next, this.lambda, this.score_max,
                            this.allPredicates, this.allPredicateSize);
                }
            }

            logger.info(">>>> finish generate, aggregate, and prune Next Lattice, using time: {}", (System.currentTimeMillis() - start) / 1000);

            logger.info("#### nextLattice size: {}", nextLattice.size());

//            logger.info("----- nextLattice");
//            for (LatticeVertex lv : nextLattice.getLatticeLevel().values()) {
//                logger.info("#### {}", lv.printCurrent());
//            }

            return nextLattice;
        }

        // broadcast necessary information
        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());

        BroadcastLattice broadcastLattice = new BroadcastLattice(this.allPredicates, this.invalidX,
                this.invalidXRHSs, this.validREEsMap, this.validLatticeVertexMap, predicateProviderIndex,
                this.Q_next, this.ifPrune, this.if_top_k, this.iteration);

        Broadcast<BroadcastLattice> bcpsLattice = sc.broadcast(broadcastLattice);

        PredicateSetAssist psAssist = new PredicateSetAssist();
        psAssist.setIndexProvider(PredicateSet.indexProvider);//在这里设置，这个值肯定是确定了的
        psAssist.setBf(PredicateSet.bf);
        Broadcast<PredicateSetAssist> bcpsAssist = sc.broadcast(psAssist);

        // split work units
        ArrayList<Lattice> workUnitLattices;
        if (this.iteration == 0) {  // (1) top-k, or (2) the first iteration of top-k diversified
            workUnitLattices = lattice.splitLattice(NUM_LATTICE);
        } else {
            workUnitLattices = lattice.splitLatticeNew(NUM_LATTICE);
        }

        if (workUnitLattices.isEmpty()) {
            return null;
        }

        long start = System.currentTimeMillis();

        // parallel expand lattice vertices of next level.
        Lattice nextLattice;
        // Use different aggregate functions!
        if (this.iteration == 0) {
            nextLattice = sc.parallelize(workUnitLattices, workUnitLattices.size()).map(task -> {

                // synchronized (BroadcastLattice.class) {
                // get IndexProvider instances
                PredicateSetAssist assist = bcpsAssist.getValue();
                PredicateSet.indexProvider = assist.getIndexProvider();
                PredicateSet.bf = assist.getBf();
                //}

                // get lattice data
                BroadcastLattice bLattice = bcpsLattice.getValue();

                Lattice latticeWorker = task.generateNextLatticeLevel(bLattice.getAllPredicates(), bLattice.getInvalidX(),
                        bLattice.getInvalidXRHSs(), bLattice.getValidREEsMap(), bLattice.getValidLatticeVertexMap(),
                        bLattice.getPredicateProviderIndex(), bLattice.getIteration(),
                        bLattice.getQ_next(), bLattice.isIfPrune());

//            // further pruning by checking UB. Note that, in this step, we can not use validREEsMap to prune! Because valid REEs may need to re-compute div score, if not pruned in work units stage.
//            if (!bLattice.isIf_top_k()) {
//                latticeWorker.pruneLatticeBeforeComputeScore(bLattice.getPrunedLocalREEs(), bLattice.isIf_UB_prune());
//            }

                return latticeWorker;

            }).aggregate(null, new ILatticeAggFunction(), new ILatticeAggFunction());
        } else {
            nextLattice = sc.parallelize(workUnitLattices, workUnitLattices.size()).map(task -> {

                // synchronized (BroadcastLattice.class) {
                // get IndexProvider instances
                PredicateSetAssist assist = bcpsAssist.getValue();
                PredicateSet.indexProvider = assist.getIndexProvider();
                PredicateSet.bf = assist.getBf();
                //}

                // get lattice data
                BroadcastLattice bLattice = bcpsLattice.getValue();

                Lattice latticeWorker = task.generateNextLatticeLevel(bLattice.getAllPredicates(), bLattice.getInvalidX(),
                        bLattice.getInvalidXRHSs(), bLattice.getValidREEsMap(), bLattice.getValidLatticeVertexMap(),
                        bLattice.getPredicateProviderIndex(), bLattice.getIteration(),
                        bLattice.getQ_next(), bLattice.isIfPrune());

//            // further pruning by checking UB. Note that, in this step, we can not use validREEsMap to prune! Because valid REEs may need to re-compute div score, if not pruned in work units stage.
//            if (!bLattice.isIf_top_k()) {
//                latticeWorker.pruneLatticeBeforeComputeScore(bLattice.getPrunedLocalREEs(), bLattice.isIfPrune());
//            }

                return latticeWorker;

            }).aggregate(null, new ILatticeAggFunctionNew(), new ILatticeAggFunctionNew());
        }

        // pruning by RHS intersection
        if (this.iteration == 0) {  // (1) top-k, or (2) the first iteration of top-k diversified
            nextLattice.removeInvalidLatticeAndRHSs(lattice);
        }
        else {
            /*
            logger.info("----- before pruneLatticeBeforeComputeScore, nextLattice:");
            for (LatticeVertex lv : nextLattice.getLatticeLevel().values()) {
                logger.info("#### {}", lv.printCurrent());
            }
            */
            // Note that, we do NOT use allLatticeVertexBits to prune, when iteration != 0
            if (this.ifPrune) {
                nextLattice.pruneLatticeBeforeComputeScore(this.intermediateREEs_past, this.Q_next, this.lambda, this.score_max,
                        this.allPredicates, this.allPredicateSize);
            }
        }

        logger.info(">>>> finish generate, aggregate, and prune Next Lattice, using time: {}", (System.currentTimeMillis() - start) / 1000);

        logger.info("#### nextLattice size: {}", nextLattice.size());

//        logger.info("----- nextLattice");
//        for (LatticeVertex lv : nextLattice.getLatticeLevel().values()) {
//            logger.info("#### {}", lv.printCurrent());
//        }

        return nextLattice;
    }


//    public List<Message> run(ArrayList<WorkUnit> workUnits, String taskId, SparkSession spark,
//                             SparkContextConfig sparkContextConfig, Broadcast<PredicateSetAssist> bcpsAssist) {
//        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
//        Broadcast<List<Predicate>> broadAllPredicate = sc.broadcast(allPredicates);
//        return this.run(workUnits, taskId, sc, bcpsAssist);
//    }


    public List<Message> run(ArrayList<WorkUnit> workUnits, String taskId,
                             JavaSparkContext sc, Broadcast<PredicateSetAssist> bcpsAssist) {
        BroadcastObj broadcastObj = new BroadcastObj(this.maxTupleNum, this.inputLight, this.support, this.confidence,
                this.maxOneRelationNum, this.tupleNumberRelations, this.relevance, this.diversity, this.sampleRatioForTupleCov, this.final_results,
                this.index_null_string, this.index_null_double, this.index_null_long);

        broadcastObj.setValidConstantRule(this.validConstantRule);
        Broadcast<BroadcastObj> scInputLight = sc.broadcast(broadcastObj);

        //增加聚类方法聚合Unit
        ArrayList<WorkUnits> unitSets;
        if (this.if_cluster_workunits == 1) {
            logger.info(">>>>begin cluster");
            unitSets = clusterByPredicate(workUnits, 2);
            logger.info(">>>>end cluster");
        } else {
            unitSets = new ArrayList<>();
            for (WorkUnit workUnit : workUnits) {
                WorkUnits cluster = new WorkUnits();
                cluster.addUnit(workUnit);
                unitSets.add(cluster);  // one unitSets contains one WorkUnit, along with one pid_
            }
        }

        List<Message> ruleMessages = new ArrayList<>();
        logger.info("running by MultiTuplesRuleMiningOpt!!");

        // set allCount of each work unit
        for(WorkUnits set : unitSets) {
            set.setAllCount(this.allCount);
        }

        List<Message> ruleMessagesSub = sc.parallelize(unitSets, unitSets.size()).map(unitSet -> {
            if (unitSet.getCurrrent().size() == 0) {
                return null;
            }

            PredicateSetAssist assist = bcpsAssist.getValue();
            PredicateSet.indexProvider = assist.getIndexProvider();
            PredicateSet.bf = assist.getBf();
            String taskid = assist.getTaskId();

            BroadcastObj bobj = scInputLight.getValue();
            Map<PredicateSet, List<Predicate>> validConsRuleMap = bobj.getValidConstantRule();
            Map<PredicateSet, Map<String, Predicate>> constantXMap = new HashMap<>();

            // discard the work units that only need diversity re-computation, when computing support-based measure
            // unitSet_filtered is checked in support-based measure, while unitSet is checked in diversity measure
            WorkUnits unitSet_filtered;
            if (!this.diversity.getCandidate_functions().contains("tuple_coverage")) {
                unitSet_filtered = new WorkUnits();
                for (WorkUnit unit : unitSet.getUnits()) {
                    if (!unit.isCompute_div_only()) {
                        unitSet_filtered.addUnit(unit);
                    }
                }
            } else {
                unitSet_filtered = new WorkUnits(unitSet);
            }

            PredicateSet sameSet = unitSet_filtered.getSameSet();
            for (PredicateSet set : validConsRuleMap.keySet()) {
                PredicateSet tupleX = new PredicateSet();
                Map<String, Predicate> constantX = new HashMap<>();
                for (Predicate p : set) {
                    if (p.isConstant()) {
                        constantX.put(p.getOperand1().toString_(0), p);
                    } else {
                        tupleX.add(p);
                    }
                }

                if (tupleX.size() > 0) {
                    constantXMap.putIfAbsent(tupleX, constantX);
                }
            }

            List<WorkUnit> units = unitSet_filtered.getUnits();
            for (PredicateSet tuplePs : constantXMap.keySet()) {
                if (sameSet.size() > 0) {
                    if (!tuplePs.containsPS(sameSet)) {
                        continue;
                    }
                }
                for (WorkUnit unit : units) {
                    PredicateSet tupleX = new PredicateSet();
                    PredicateSet constantX = new PredicateSet();
                    for (Predicate p : unit.getCurrrent()) {
                        if (p.isConstant()) {
                            constantX.add(p);
                        } else {
                            tupleX.add(p);
                        }
                    }


                    if (tupleX.containsPS(tuplePs)) {
                        Map<String, Predicate> constantsMap = constantXMap.get(tuplePs);
                        boolean iscont = true;
                        for (Predicate p : constantX) {
                            if (!constantsMap.containsKey(p.getOperand1().toString_(0))) {
                                iscont = false;
                                break;
                            }
                        }

                        if (iscont) {
                            PredicateSet lhs = new PredicateSet();
                            lhs.addAll(tuplePs);
                            for (Predicate p : constantsMap.values()) {
                                lhs.add(p);
                            }
                            List<Predicate> rhs = validConsRuleMap.get(lhs);

                            if (rhs != null) {
                                for (Predicate p : rhs) {
                                    if (unit.getRHSs().containsPredicate(p)) {
                                        unit.getRHSs().remove(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Predicate pBegin = null;
            for (Predicate p : unitSet_filtered.getSameSet()) {
                if (!p.isML() && !p.isConstant()) {
                    pBegin = p;
                    break;
                }
            }

            MultiTuplesRuleMiningOpt multiTuplesRuleMining = new MultiTuplesRuleMiningOpt(bobj.getMax_num_tuples(),
                    bobj.getInputLight(), bobj.getSupport(), bobj.getConfidence(), bobj.getMaxOneRelationNum(),
                    unitSet.getAllCount(), bobj.getTupleNumberRelations(),
                    bobj.getRelevance(), bobj.getDiversity(), bobj.getSampleRatioForTupleCov(),
                    bobj.getIndex_null_string(), bobj.getIndex_null_double(), bobj.getIndex_null_long());

            ArrayList<String> rel_funcs = bobj.getRelevance().getCandidate_functions();
            ArrayList<String> div_funcs = bobj.getDiversity().getCandidate_functions();

            List<Message> messages = new ArrayList<>();  // can not be null, null can not be serialized!
            // support-related computation
//            if (rel_funcs.contains("support") || rel_funcs.contains("confidence") || div_funcs.contains("tuple_coverage")) {  // we always need to compute the support and confidence, since we return "valid" rules.
                if (!unitSet_filtered.getUnits().isEmpty()) {
                    messages = multiTuplesRuleMining.validationMap1(unitSet_filtered, pBegin, bobj.getPartial_solution()); // when if_cluster_workunits = 0, messages.size() is 1
                }
//            }

            if (div_funcs.isEmpty()) { // "null"
                return messages;
            }

            if (div_funcs.size() == 1 && div_funcs.contains("tuple_coverage")) {
                return messages;
            }

            // data is split to many slices, but we only compute diversity once. on pid 0.
            int[] pids = unitSet.getPids();
            boolean compute_div = true;
            for (int i = 0; i < pids.length; i++) {
                if (pids[i] != 0) {
                    compute_div = false;
                }
            }
            if (!compute_div) {
                return messages;
            }

            // diversity computation
            List<Message> messages_div = new ArrayList<>();  // when if_cluster_workunits = 0, messages_div.size() is 1
            if (this.final_results.isEmpty()) {
                messages_div = multiTuplesRuleMining.initialZeroDiversity(unitSet);
            } else {
                if (div_funcs.contains("attribute_nonoverlap")) {
                    messages_div = multiTuplesRuleMining.computeAttributeNonoverlap(unitSet);
                }
                if (div_funcs.contains("predicate_nonoverlap")) {
                    List<Message> messages_temp = multiTuplesRuleMining.computePredicateNonoverlap(unitSet);
                    if (messages_div.isEmpty()) {
                        messages_div = messages_temp;
                    } else {
                        for (Message msg : messages_div) {
                            msg.mergeMessageWithOtherMeasures(messages_temp);
                        }
                    }
                }
                if (div_funcs.contains("attribute_distance")) {
                    List<Message> messages_temp = multiTuplesRuleMining.computeMinAttributeDistance(unitSet, bobj.getPartial_solution());
                    if (messages_div.isEmpty()) {
                        messages_div = messages_temp;
                    } else {
                        for (Message msg : messages_div) {
                            msg.mergeMessageWithOtherMeasures(messages_temp);
                        }
                    }
                }
            }

            // merge relevance and diversity information
            for (Message msg : messages_div) {
                msg.mergeMessageBeforeSend(messages);
            }

            return messages_div;
        }).aggregate(null, new IMessageAggFunction(), new IMessageAggFunction());

        ruleMessages.addAll(ruleMessagesSub);

        return ruleMessages;
    }

    public ArrayList<WorkUnits> clusterByPredicate(List<WorkUnit> units, int clusterSize) {
        List<WorkUnits> clusters = new ArrayList<>();
        List<WorkUnit> workUnits = new ArrayList<>();
        int minSize = 0;
        Map<String, List<WorkUnit>> workUnitMap = new HashMap<>();
        Map<IBitSet, List<WorkUnit>> sameUnitMap = new HashMap<>();
        for (WorkUnit unit : units) {
            workUnitMap.putIfAbsent(unit.getPidString(), new ArrayList<>());
            workUnitMap.get(unit.getPidString()).add(unit);
            minSize = Math.max(unit.getCurrrent().size(), minSize);

            sameUnitMap.putIfAbsent(unit.getKey(), new ArrayList<>());
            sameUnitMap.get(unit.getKey()).add(unit);
        }

        for (List<WorkUnit> list : sameUnitMap.values()) {
            list.sort(new Comparator<WorkUnit>() {
                @Override
                public int compare(WorkUnit o1, WorkUnit o2) {
                    int[] pids1 = o1.getPids();
                    int[] pids2 = o2.getPids();
                    for (int index = 0; index < pids1.length; index++) {
                        if (pids1[index] != pids2[index]) {
                            return pids1[index] - pids2[index];
                        }
                    }
                    return 0;
                }
            });
        }

        minSize = Math.max(minSize + 1, 0);
//        minSize = Math.max((minSize * 2) - 3, 1);
//        for(Predicate p : this.allPredicates) {
//            WorkUnits cluster = new WorkUnits();
//            cluster.addSameSet(p);
//            size++;
//            if (size == clusterSize) {
//                break;
//            }
//        }

        for (List<WorkUnit> sameUnit : workUnitMap.values()) {
            workUnits.addAll(sameUnit);

            int totalSize = Math.min(clusterSize, units.size());
            for (int size = 0; size < totalSize; size++) {
                WorkUnits cluster = new WorkUnits();
                cluster.addUnit(units.get(size));
                clusters.add(cluster);
                workUnits.remove(units.get(size));
            }

            for (WorkUnit unit : workUnits) {
                int max = minSize;
                int current = -1;
                for (int index = 0; index < clusters.size(); index++) {
                    WorkUnits cluster = clusters.get(index);
                    int sameNum = cluster.calDistance(unit);
                    if (sameNum > max) {
                        max = sameNum;
                        current = index;
                    }
                }

                if (current > -1) {
                    WorkUnits cluster = clusters.get(current);
                    cluster.addUnit(unit);
                } else {
                    WorkUnits cluster = new WorkUnits();
                    cluster.addUnit(unit);
                    clusters.add(cluster);
                }
            }

            workUnits.clear();
            break;
        }
        ArrayList<WorkUnits> result = new ArrayList<>();
        for (WorkUnits cluster : clusters) {
            if (cluster.getUnits().size() > 0) {

                List<WorkUnits> list = new ArrayList<>();
                for (WorkUnit unit : cluster.getUnits()) {
                    int count = 0;
                    for (WorkUnit allUnit : sameUnitMap.get(unit.getKey())) {
                        if (list.size() == count) {
                            WorkUnits newCluster = new WorkUnits();
                            newCluster.addUnit(allUnit);
                            list.add(newCluster);
                        } else {
                            list.get(count).addUnit(allUnit);
                        }
                        count++;
                    }
                }
                result.addAll(list);
            }
        }

//        for (WorkUnits cluster : result) {
//            logger.info(">>>>show unit size : {} | same set: {}", cluster.getUnits().size(), cluster.getSameSet());
//        }
        return result;
    }


    private void prepareAllPredicatesMultiTuples() {
        HashMap<String, ParsedColumnLight<?>> colsMap = new HashMap<>();
        for (Predicate p : this.allPredicates) {
            String k = p.getOperand1().getColumn().toStringData();
            if (!colsMap.containsKey(k)) {
                ParsedColumnLight<?> col = new ParsedColumnLight<>(p.getOperand1().getColumn(), p.getOperand1().getColumn().getType());
                colsMap.put(k, col);
            }
            k = p.getOperand2().getColumn().toStringData();
            if (!colsMap.containsKey(k)) {
                ParsedColumnLight<?> col = new ParsedColumnLight<>(p.getOperand2().getColumn(), p.getOperand2().getColumn().getType());
                colsMap.put(k, col);
            }
        }
        // delete value int data of ParsedColumn
        for (Predicate p : this.allPredicates) {
            p.getOperand1().getColumn().cleanValueIntBeforeBroadCast();
            p.getOperand2().getColumn().cleanValueIntBeforeBroadCast();
        }
        // insert parsedColumnLight for each predicate
        PredicateSet ps = new PredicateSet();
        for (Predicate p : this.allPredicates) {
            // set columnLight
            String k = p.getOperand1().getColumn().toStringData();
            p.getOperand1().setColumnLight(colsMap.get(k));
            k = p.getOperand2().getColumn().toStringData();
            p.getOperand2().setColumnLight(colsMap.get(k));
            ps.add(p);
            for (int t1 = 0; t1 < maxTupleNum; t1++) {
                if (p.isConstant()) {
                    Predicate p_new = predicateProviderIndex.getPredicate(p, t1, t1);
                    k = p_new.getOperand1().getColumn().toStringData();
                    p_new.getOperand1().setColumnLight(colsMap.get(k));
                    k = p_new.getOperand2().getColumn().toStringData();
                    p_new.getOperand2().setColumnLight(colsMap.get(k));
                    ps.add(p_new);
                    continue;
                }
                for (int t2 = t1 + 1; t2 < maxTupleNum; t2++) {
                    Predicate p_new = predicateProviderIndex.getPredicate(p, t1, t2);
                    k = p_new.getOperand1().getColumn().toStringData();
                    p_new.getOperand1().setColumnLight(colsMap.get(k));
                    k = p_new.getOperand2().getColumn().toStringData();
                    p_new.getOperand2().setColumnLight(colsMap.get(k));
                    ps.add(p_new);
                }
            }
        }

        this.allPredicateSize = ps.getAllPredicateSize();
    }

    private void addInvalidX(List<Message> messages) {
        if (messages == null) {
            return;
        }
        for (Message message : messages) {
            if (message.isCompute_div_only()) {
                // for the valid/intermediate rules that only need re-computing the div score, the supp is 0 in the message, and will be added into invalidX by mistake!
                continue;
            }
            PredicateSet kkps = message.getCurrentSet();
//            logger.info("X: {}, supp: {}", message.getCurrentSet(), message.getCurrentSupp());
            if (message.getCurrentSupp() < this.support) {
                this.invalidX.add(kkps.getBitset());
//                logger.info("add invalidX-1: {}, supp: {} | {}", kkps, message.getCurrentSupp(), this.support);
            }
            for (Predicate invalidRHS : message.getInvalidRHSs()) {
                PredicateSet temp = new PredicateSet(kkps);
                temp.add(invalidRHS);
                this.invalidX.add(temp.getBitset());
//                logger.info("add invalidX-2: {}", temp);
                // add invalid RHS that should be removed
                this.invalidXRHSs.putIfAbsent(kkps.getBitset(), new HashSet<>());
                this.invalidXRHSs.get(kkps.getBitset()).add(invalidRHS);
//                logger.info("add invalidXRHS, X: {} -> Y: {}, supp: {} | {}", kkps, invalidRHS, message.getAllCurrentRHSsSupportOrigin().get(PredicateSet.getIndex(invalidRHS)), this.support);
            }
        }
    }

    // given a bitset, return the max index of true value
    public int get_max_index(IBitSet bs) {
        int index = bs.nextSetBit(0); // the first index of true value
        if (index >= 0 && index < this.allPredicateSize) {
            index = bs.nextSetBit(index + 1);
        }
        return index;
    }

    public void updateQnextAndIntermediateREEsPast() {
        // collect REEs that should be pruned in lattice
        for (Map.Entry<Integer, HashMap<IBitSet, HashMap<Predicate, HashMap<String, Double>>>> entry1 : this.intermediateREEs_current.entrySet()) {
            int Xset_size = entry1.getKey();
//            this.intermediateREEs_past.putIfAbsent(Xset_size, new HashMap<>());
            for (Map.Entry<IBitSet, HashMap<Predicate, HashMap<String, Double>>> entry2 : entry1.getValue().entrySet()) {
                IBitSet Xset = entry2.getKey();
//                int index_Xset = this.get_max_index(Xset);
//                this.intermediateREEs_past.get(Xset_size).putIfAbsent(Xset, new HashMap<>());
                for (Map.Entry<Predicate, HashMap<String, Double>> entry_ : entry2.getValue().entrySet()) {
                    Predicate rhs = entry_.getKey();
//                    int index_rhs = PredicateSet.getIndex(rhs);
                    HashMap<String, Double> info_ = entry_.getValue();
//                    // add to intermediateREEs_past
//                    this.intermediateREEs_past.get(Xset_size).get(Xset).put(rhs, info_);
                    // pruning the ones with UB score <= score_max
                    if (info_.get("UB") <= this.score_max) {
                        this.Q_next.putIfAbsent(Xset_size, new HashMap<>());
                        this.Q_next.get(Xset_size).putIfAbsent(Xset, new HashMap<>());
                        this.Q_next.get(Xset_size).get(Xset).put(rhs, info_);
//                        logger.info("#### Q_next, X: {}, Y: {}", new PredicateSet(Xset), rhs);

                        /*
//                        // check whether Xset and rhs contain one of the {t0.A=t1.B, t0.A='v1', t1.B='v2'}
//                        ArrayList<Boolean> isExist = this.checkTrivialExtension(new PredicateSet(Xset), null, rhs);
//                        boolean t0_exist = isExist.get(0);
//                        boolean t1_exist = isExist.get(1);
//                        boolean t0_t1_exist = isExist.get(2);
//                        boolean isTrivial = (t0_exist && t1_exist) || (t0_exist && t0_t1_exist) || (t1_exist && t0_t1_exist);
//                        int index_check = -1;
//                        if (isTrivial) {
//                            if (!t0_exist) {
//                                index_check = 0;
//                            } else if (!t1_exist) {
//                                index_check = 1;
//                            } else if (!t0_t1_exist){
//                                index_check = 2;
//                            }
//                        }
//                        String attr_trivial = rhs.getOperand1().getColumnLight().getName();

                        // add the nodes that Xset -> rhs can not be expanded to, due to only add predicate with larger index in Lattice.generateNextLatticeLevel().
                        for (Predicate pred : this.allPredicates) {
                            if (pred.equals(rhs)) {
                                continue;
                            }
                            PredicateSet temp = new PredicateSet(Xset);
                            if (temp.containsPredicate(pred)) {
                                continue;
                            }
                            int index_temp = Math.max(index_Xset, index_rhs);
                            if (PredicateSet.getIndex(pred) >= index_temp) {
                                break;
                            }
                            if (this.checkTrivialExtension(temp, rhs, pred)) {
                                continue;
                            }
//                            if (isTrivial && pred.getOperand1().getColumnLight().getName().equals(attr_trivial)) {
//                                if (index_check == -1) {
//                                    continue;
//                                }
//                                if (pred.isConstant()) {
//                                    if (pred.getIndex1() == index_check) {
//                                        continue;
//                                    }
//                                } else {
//                                    if (index_check == 2) {
//                                        continue;
//                                    }
//                                }
//                            }
                            temp.add(pred);
                            int temp_size = Xset_size + 1;
                            this.Q_next.putIfAbsent(temp_size, new HashMap<>());
                            this.Q_next.get(temp_size).putIfAbsent(temp.getBitset(), new HashMap<>());
                            this.Q_next.get(temp_size).get(temp.getBitset()).put(rhs, null);
                        }
                        */
                    }
                }
            }
        }
    }

    public boolean checkTrivialExtension(PredicateSet Xset, Predicate rhs, Predicate newP) {
        PredicateSet lv = new PredicateSet(Xset);
        if (rhs != null) {
            lv.add(rhs);
        }
        boolean t0_exist = false;
        boolean t1_exist = false;
        boolean t0_t1_exist = false;
        if (newP.isConstant()) {
            String attr = newP.getOperand1().getColumnLight().getName();
            int index = newP.getIndex1();
            if (index == 0) {
                t0_exist = true;
            } else {
                t1_exist = true;
            }
            for (Predicate p : lv) {
                if (p.isConstant()) {
                    if (!p.getOperand1().getColumnLight().getName().equals(attr)) {
                        continue;
                    }
                    int index_p = p.getIndex1();
                    if (index_p == 0) {
                        t0_exist = true;
                    } else {
                        t1_exist = true;
                    }
                } else {
                    if (!p.getOperand1().getColumnLight().getName().equals(attr)) {
                        continue;
                    }
                    if (!p.getOperand2().getColumnLight().getName().equals(attr)) {
                        continue;
                    }
                    t0_t1_exist = true;
                }
            }
        } else {
            t0_t1_exist = true;
            String attr_1 = newP.getOperand1().getColumnLight().getName();
            String attr_2 = newP.getOperand2().getColumnLight().getName();
            for (Predicate p : lv) {
                if (p.isConstant()) {
                    int index_p = p.getIndex1();
                    if (index_p == newP.getIndex1() && p.getOperand1().getColumnLight().getName().equals(attr_1)) {
                        if (index_p == 0) {
                            t0_exist = true;
                        } else {
                            t1_exist = true;
                        }
                    }
                    if (index_p == newP.getIndex2() && p.getOperand1().getColumnLight().getName().equals(attr_2)) {
                        if (index_p == 0) {
                            t0_exist = true;
                        } else {
                            t1_exist = true;
                        }
                    }
                }
            }
        }

        return t0_exist && t1_exist && t0_t1_exist;

//        ArrayList<Boolean> res = new ArrayList<>();
//        res.add(t0_exist);
//        res.add(t1_exist);
//        res.add(t0_t1_exist);
//        return res;

    }

    private void getCandidateXSupportRatios(HashMap<PredicateSet, Double> results, List<Message> messages) {
        if (messages == null) {
            return;
        }
        for (Message message : messages) {
            if (message.isCompute_div_only()) {
                continue;
            }
            PredicateSet kkps = message.getCurrentSet(); //new PredicateSet();
            if (!results.containsKey(kkps)) {
                // results.put(kkps, message.getCurrentSupp() * 1.0 / this.allCount);
                results.put(kkps, message.getCurrentSupp() * 1.0 / this.allCount);
            }

            // consider X + p_0 in message, check their interestingness upper bound later.
            for (Map.Entry<Predicate, Long> entry : message.getAllCurrentRHSsSupport().entrySet()) {
                Predicate rhs = entry.getKey();
                if (message.getValidRHSs().contains(rhs) || message.getInvalidRHSs().contains(rhs)) {
                    continue;
                }
                PredicateSet kkkps = new PredicateSet(kkps);
                kkkps.add(rhs);

                if (!results.containsKey(kkkps)) {
                    if (rhs.isConstant()) {
                        results.put(kkkps, entry.getValue() * maxOneRelationNum * 1.0 / this.allCount);
                    } else {
                        results.put(kkkps, entry.getValue() * 1.0 / this.allCount);
                    }
                }
            }
        }
    }

    // prune work units that are intermediate rules
    public ArrayList<WorkUnit> pruneWorkUnits(ArrayList<WorkUnit> workUnits) {
        ArrayList<WorkUnit> workUnits_new = new ArrayList<>();

        for (WorkUnit task : workUnits) {
            IBitSet t = task.getCurrrent().getBitset();
            PredicateSet removeRHSs = new PredicateSet();
            int Xset_size = task.getCurrrent().size();
            // remove rhs due to UB_lazy <= this.score_max
            if (this.intermediateREEs_past.containsKey(Xset_size)) {
                HashMap<Predicate, HashMap<String, Double>> rhs_info_ = this.intermediateREEs_past.get(Xset_size).get(t);
                if (rhs_info_ != null) {
                    for (Predicate rhs : task.getRHSs()) {
                        HashMap<String, Double> info_ = rhs_info_.get(rhs);
                        if (info_ != null && info_.get("UB_lazy") <= this.score_max) {
                            removeRHSs.add(rhs);
                        }
                    }
                }
            }
            for (Predicate p : removeRHSs) {
                task.removeRHS(p);
//                logger.info("#### pruneWorkUnits, remove {}", task);
            }
            if (task.getRHSs().size() > 0) {
                workUnits_new.add(task);
            }
        }

        return workUnits_new;
    }


    /**
     * deal with skewness
     * split heavy work units
     */
    public int solveSkewness(ArrayList<WorkUnit> tasks, ArrayList<WorkUnit> tasks_new) {
        HashMap<String, ArrayList<int[]>> recordComs = new HashMap<>();
        int[] _pids = new int[this.maxTupleNum + 1];
        String[] keys = new String[this.maxTupleNum + 1];
        int wid = 0;
        for (wid = 0; wid < tasks.size(); wid++) {
            if (tasks_new.size() > this.MIN_NUM_WORK_UNITS) {
                break;
            }
            WorkUnit task = tasks.get(wid);
            if (task.getRHSs().size() <= 0) {
                continue;
            }

            for (int i = 0; i < _pids.length; i++) _pids[i] = 0;

            if (task.isCompute_div_only()) {  // (1) task is from Q_past, i.e., the valid rules and (2) diversity do not contains tuple_coverage
                WorkUnit task_new = new WorkUnit(task, _pids);
                tasks_new.add(task_new);
                continue;
            }

            // for the intermediate rules that only need to re-compute the diversity w.r.t the current partial solution
            if (this.iteration != 0 && !this.diversity.getCandidate_functions().contains("tuple_coverage") &&
                    this.intermediateREEs_past.containsKey(task.getCurrrent().size())) {
                ArrayList<Predicate> rhs_div_only = new ArrayList<>();
                HashMap<Predicate, HashMap<String, Double>> rhs_info_ = this.intermediateREEs_past.get(task.getCurrrent().size()).get(task.getCurrrent().getBitset());
                if (rhs_info_ != null) {
                    for (Predicate p : task.getRHSs()) {
                        if (rhs_info_.containsKey(p)) {
                            rhs_div_only.add(p);
                        }
                    }
                }
                if (!rhs_div_only.isEmpty()) {
                    WorkUnit task_new = new WorkUnit(task, rhs_div_only, _pids);
                    tasks_new.add(task_new);

                    for (Predicate p : rhs_div_only) {
                        task.removeRHS(p);
                    }
                }
            }

            if (task.getRHSs().size() <= 0) {
                continue;
            }

            // collect the number of pids for each tid
            int tupleNum = 0;
            // only consider current, because RHS is <t_0, t_1>
            for (Predicate p : task.getCurrrent()) {
                // index1 : pid
                tupleNum = Math.max(tupleNum, p.getIndex1());
                int numOfPid = p.getOperand1().getColumn().getPliSections().size();
                _pids[p.getIndex1()] = numOfPid;
                keys[p.getIndex1()] = p.getOperand1().getColumn().getTableName();
                tupleNum = Math.max(tupleNum, p.getIndex2());
                _pids[p.getIndex2()] = p.getOperand2().getColumn().getPliSections().size();
                keys[p.getIndex2()] = p.getOperand2().getColumn().getTableName();
            }
            String comKey = "";
            for (int i = 0; i <= tupleNum; i++) comKey += keys[i] + "-";
            if (!recordComs.containsKey(comKey)) {
                // enumerate all combinations of pids
                ArrayList<int[]> combinations = new ArrayList<>();
//                int[] oneCom = new int[tupleNum + 1];
                int[] oneCom = new int[this.maxTupleNum + 1];
                enumCombinations(oneCom, combinations, _pids, 0, tupleNum + 1);
                recordComs.put(comKey, combinations);
            }
            // copy work units, with different combinations of pids
            ArrayList<int[]> allCombinations = recordComs.get(comKey);
            for (int[] combination : allCombinations) {
                WorkUnit task_new = new WorkUnit(task, combination);
//                task_new.setTransferData();
                tasks_new.add(task_new);
            }
        }

        return wid;
    }

    private void enumCombinations(int[] currentPIDs, ArrayList<int[]> results, int[] _pids, int script, int tupleNum) {
        if (script >= tupleNum) {
            int[] com = currentPIDs.clone();
            results.add(com);
            return;
        }
        for (int i = 0; i < _pids[script]; i++) {
            currentPIDs[script] = i;
            enumCombinations(currentPIDs, results, _pids, script + 1, tupleNum);

        }
    }

    // split work units according to their RHSs according to different types of tuple ID
    public ArrayList<WorkUnit> partitionWorkUnitsByRHSs(ArrayList<WorkUnit> tasks) {
        ArrayList<WorkUnit> tasks_new = new ArrayList<>();
        for (int i = 0; i < tasks.size(); i++) {
            WorkUnit task = tasks.get(i);
            // split RHS predicates
            ArrayList<PredicateSet> rhss_group = this.splitRHSsByTIDs(task.getRHSs());
            for (PredicateSet subRHSs : rhss_group) {
                WorkUnit subTask = new WorkUnit(task);
                subTask.resetRHSs(subRHSs);
                tasks_new.add(subTask);
            }
        }
        return tasks_new;
    }

    private ArrayList<PredicateSet> splitRHSsByTIDs(PredicateSet ps) {
        ArrayList<PredicateSet> res = new ArrayList<>();
        HashMap<Integer, PredicateSet> tempMap = new HashMap<>();
        for (Predicate p : ps) {
            int t_0 = p.getIndex1();
            int t_1 = p.getIndex2();
            int key = MultiTuplesOrder.encode(new ImmutablePair<>(t_0, t_1));
            if (tempMap.containsKey(key)) {
                tempMap.get(key).add(p);
            } else {
                PredicateSet tempps = new PredicateSet();
                tempps.add(p);
                tempMap.put(key, tempps);
            }
        }
        for (Map.Entry<Integer, PredicateSet> entry : tempMap.entrySet()) {
            res.add(entry.getValue());
        }
        return res;
    }


    // integrate messages
    List<Message> integrateMessages(List<Message> messages) {
        List<Message> messages_new = new ArrayList<>();
        HashMap<PredicateSet, ArrayList<Message>> messageGroups = new HashMap<>();
        // cluster the messages with the same X
        for (Message message : messages) {
            message.transformCurrent();
            PredicateSet key = message.getCurrentSet();
            if (messageGroups.containsKey(key)) {
                messageGroups.get(key).add(message);
            } else {
                ArrayList<Message> tt = new ArrayList<>();
                tt.add(message);
                messageGroups.put(key, tt);
            }
        }

        // integration
        ArrayList<String> rel_funcs = this.relevance.getCandidate_functions();
        ArrayList<String> div_funcs = this.diversity.getCandidate_functions();
        for (Map.Entry<PredicateSet, ArrayList<Message>> entry : messageGroups.entrySet()) {
            Message mesg = entry.getValue().get(0);
            for (int i = 1; i < entry.getValue().size(); i++) {
                mesg.mergeMessage(entry.getValue().get(i), rel_funcs, div_funcs);
            }
            if (rel_funcs.contains("support") || rel_funcs.contains("confidence") || div_funcs.contains("tuple_coverage")) {
                mesg.updateMessage(this.support, this.confidence, this.maxOneRelationNum);
            }
            messages_new.add(mesg);
        }
        return messages_new;
    }

    // integrate messages. work units with the same X, may different from Y, since some work units only compute diversity.
    List<Message> integrateMessages_new(List<Message> messages) {
        List<Message> messages_new = new ArrayList<>();
        HashMap<PredicateSet, HashMap<PredicateSet, ArrayList<Message>>> messageGroups = new HashMap<>();
        // cluster the messages with the same X and the same Y
        for (Message message : messages) {
            if (message.isEmpty()) {
                continue;
            }
            message.transformCurrent();
            PredicateSet Xset = message.getCurrentSet();
            PredicateSet rhsList = message.getRHSList();
            messageGroups.putIfAbsent(Xset, new HashMap<>());
            messageGroups.get(Xset).putIfAbsent(rhsList, new ArrayList<>());
            messageGroups.get(Xset).get(rhsList).add(message);
        }

        // integration
        ArrayList<String> rel_funcs = this.relevance.getCandidate_functions();
        ArrayList<String> div_funcs = this.diversity.getCandidate_functions();
        for (Map.Entry<PredicateSet, HashMap<PredicateSet, ArrayList<Message>>> entry : messageGroups.entrySet()) {
            for (Map.Entry<PredicateSet, ArrayList<Message>> entry_ : entry.getValue().entrySet()) {
                Message mesg = entry_.getValue().get(0);
                for (int i = 1; i < entry_.getValue().size(); i++) {
                    mesg.mergeMessage(entry_.getValue().get(i), rel_funcs, div_funcs);
                }
                if (!mesg.isCompute_div_only()) {
//                    if (rel_funcs.contains("support") || rel_funcs.contains("confidence") || div_funcs.contains("tuple_coverage")) {
                        mesg.updateMessage(this.support, this.confidence, this.maxOneRelationNum);
//                    }
                }
                /*
                logger.info("#### ------------------");
                logger.info("#### X:{}", mesg.getCurrentSet());
                logger.info("#### supp:{}, suppXC0: {}, suppXC1: {}", mesg.getCurrentSupp(), mesg.getCurrentSuppCP0(), mesg.getCurrentSuppCP1());
                for (Map.Entry<Predicate, Long> entry1 : mesg.getAllCurrentRHSsSupport().entrySet()) {
                    logger.info("#### Y: {}, supp: {}", entry1.getKey(), entry1.getValue());
                }
                */
                messages_new.add(mesg);
            }
        }

        return messages_new;
    }

}
