package sics.seiois.mlsserver.biz.der.metanome;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.PredicateSetTableMsg;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;
import sics.seiois.mlsserver.utils.MlsConstant;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class REE implements Serializable {
    private static final long serialVersionUID = 1979294198571833701L;

    private static final Logger logger = LoggerFactory.getLogger(REE.class);
    private PredicateSet currentList;
    private Predicate rhs;
    private long lhsSupport;
    private long support;
    private long violations;
    private double confidence;

    private int length;

    private double relevance_score;
    private double diversity_score;
    private double score;

    private double supp_score;
    private double attr_nonoverlap_score;
    private double pred_nonoverlap_score;
    private double tuple_cov_score;
    private double attr_distance_score;

    private HashSet<String> covered_attrs;

    private Map<String, Set<Predicate>> FKMap;
    private static final int ONLY_ONE = 1;


    public REE(PredicateSet Xset, Predicate rhs) {
        this.currentList = Xset;
        this.rhs = rhs;
        this.updateCoveredAttrs();
        this.length = this.currentList.size() + 1;
    }

    public REE(PredicateSet Xset, Predicate rhs, long lhsSupport, long support, long violations, double confidence,
               double relevance_score, double diversity_score, double score,
               double supp_score, double attr_nonoverlap_score, double pred_nonoverlap_score, double tuple_cov_score,
               double attr_distance_score) {
        this.currentList = Xset;
        this.rhs = rhs;
        this.lhsSupport = lhsSupport;
        this.support = support;
        this.violations = violations;
        this.confidence = confidence;
        this.relevance_score = relevance_score;
        this.diversity_score = diversity_score;
        this.score = score;
        this.supp_score = supp_score;
        this.attr_nonoverlap_score = attr_nonoverlap_score;
        this.pred_nonoverlap_score = pred_nonoverlap_score;
        this.tuple_cov_score = tuple_cov_score;
        this.attr_distance_score = attr_distance_score;
        this.updateCoveredAttrs();
        this.length = this.currentList.size() + 1;
    }

    public REE(PredicateSet Xset, Predicate rhs,
               long lhsSupport, long support, long violations, double confidence,
               double relevance_score) {
        this.currentList = Xset;
        this.rhs = rhs;
        this.lhsSupport = lhsSupport;
        this.support = support;
        this.violations = violations;
        this.confidence = confidence;
        this.relevance_score = relevance_score;
        this.updateCoveredAttrs();
        this.length = this.currentList.size() + 1;
    }

    public void updateCoveredAttrs() {
        this.covered_attrs = new HashSet<>();
        for (Predicate p : this.currentList) {
            this.covered_attrs.add("X_" + p.getOperand1().getColumnLight().getName());
            if (!p.isConstant()) {
                this.covered_attrs.add("X_" + p.getOperand2().getColumnLight().getName());
            }
        }
        this.covered_attrs.add("Y_" + this.rhs.getOperand1().getColumnLight().getName());
        if (!this.rhs.isConstant()) {
            this.covered_attrs.add("Y_" + this.rhs.getOperand2().getColumnLight().getName());
        }
    }

    public HashSet<String> getCovered_attrs() {
        return this.covered_attrs;
    }

    public double getAttr_distance_score() {
        return attr_distance_score;
    }

    public void setLHSSupport(long lhsSupport) {
        this.lhsSupport = lhsSupport;
    }

    public void setSupport(long support) {
        this.support = support;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public void setViolations(long violations) {
        this.violations = violations;
    }

    public void setAttr_nonoverlap_score(double attr_nonoverlap_score) {
        this.attr_nonoverlap_score = attr_nonoverlap_score;
    }

    public void setPred_nonoverlap_score(double pred_nonoverlap_score) {
        this.pred_nonoverlap_score = pred_nonoverlap_score;
    }

    public void setTuple_cov_score(double tuple_cov_score) {
        this.tuple_cov_score = tuple_cov_score;
    }

    public void setAttr_distance_score(double attr_distance_score) {
        this.attr_distance_score = attr_distance_score;
    }

    public void setRelevanceScore(double relevance_score) {
        this.relevance_score = relevance_score;
    }

    public void setDiversityScore(double diversity_score) {
        this.diversity_score = diversity_score;
    }

    public void setScore(double score) {
        this.score = score;
    }


    public PredicateSet getCurrentList() {
        return this.currentList;
    }

    public Predicate getRHS() {
        return this.rhs;
    }

    public long getSupport() {
        return this.support;
    }

    public long getViolations() {
        return this.violations;
    }

    public double getConfidence() {
        return this.confidence;
    }

    public int getLength() {
        return this.length;
    }

    public double getRelevance_score() {
        return this.relevance_score;
    }

    public double getDiversity_score() {
        return this.diversity_score;
    }

    public double getScore() {
        return this.score;
    }

    public double getSupp_score() {
        return this.supp_score;
    }

    public double getAttr_nonoverlap_score() {
        return this.attr_nonoverlap_score;
    }

    public double getPred_nonoverlap_score() {
        return this.pred_nonoverlap_score;
    }

    public double getTuple_cov_score() {
        return this.tuple_cov_score;
    }

    public double computeAttributeJaccardDistance(REE ree) {
        // intersection
        HashSet<String> intersection = new HashSet<>();
        intersection.addAll(this.covered_attrs);
        intersection.retainAll(ree.getCovered_attrs());

        // union
        HashSet<String> union = new HashSet<>();
        union.addAll(this.covered_attrs);
        union.addAll(ree.getCovered_attrs());

        // Jaccard Distance
        double Jaccard_distance = 1 - intersection.size() * 1.0 / union.size();

        return Jaccard_distance;
    }

    public double computeAttributeJaccardDistance(ArrayList<Predicate> Xset, Predicate rhs) {
        // attrs in another ree
        HashSet<String> attrs_another_ree = new HashSet<>();
        for (Predicate p : Xset) {
            attrs_another_ree.add("X_" + p.getOperand1().getColumnLight().getName());
            if (!p.isConstant()) {
                attrs_another_ree.add("X_" + p.getOperand2().getColumnLight().getName());
            }
        }
        attrs_another_ree.add("Y_" + rhs.getOperand1().getColumnLight().getName());
        if (rhs.isConstant()) {
            attrs_another_ree.add("Y_" + rhs.getOperand2().getColumnLight().getName());
        }

        // intersection
        HashSet<String> intersection = new HashSet<>();
        intersection.addAll(this.covered_attrs);
        intersection.retainAll(attrs_another_ree);

        // union
        HashSet<String> union = new HashSet<>();
        union.addAll(this.covered_attrs);
        union.addAll(attrs_another_ree);

        // Jaccard Distance
        double Jaccard_distance = 1 - intersection.size() * 1.0 / union.size();

        return Jaccard_distance;
    }

    public String toString() {
        String str = "";
        for (Predicate p : this.currentList) {
            str += p.toString();
            str += " ^";
        }
        str = str.substring(0, str.length()-1);
        str += "->";
        str += this.rhs.toString();
        return str.trim();
    }

    /*
    * This toREEString() function has errors!!!
    * The output exists: t0.attr = t0.attr
    * */
    public String toREEString() {
        StringBuffer tables = new StringBuffer();
        StringBuffer joinCol = new StringBuffer();

        Map<String, PredicateSetTableMsg> tablMsg = currentList.getColumnNameStr();
        boolean isPair = false;
        for(PredicateSetTableMsg msg : tablMsg.values()) {
            if(ONLY_ONE < msg.getTableAlias().size()) {
                isPair = true;
                break;
            }
        }

        Map<String, PredicateSetTableMsg> colMap = generateFkMsg(tablMsg);

        StringBuffer reeStr = new StringBuffer(currentList.toREEString())
                .append(" -> ").append(rhs.toREEString());
        for (Map.Entry<String, PredicateSetTableMsg> table : colMap.entrySet()) {
            for(Integer alias : table.getValue().getTableAlias()) {
                tables.append(table.getKey()).append("(t").append(alias).append(") ^ ");
            }
            if(null != table.getValue().getJoinSet()) {
                for (Predicate fk : table.getValue().getJoinSet()) {
                    boolean setIncludeFk = false;
                    for(Predicate p :currentList){
                        if(p.equals(fk)) {
                            setIncludeFk = true;
                            break;
                        }
                    }
                    if(!setIncludeFk) {
                        fk.setIndex_new_1(fk.getIndex1());
                        fk.setIndex_new_2(fk.getIndex2());
                        joinCol.append(fk.toREEString()).append(" ^ ");
                    }
                    if(isPair) {
                        fk.setIndex_new_1(fk.getIndex1() + 1);
                        fk.setIndex_new_2(fk.getIndex2() + 1);
                        joinCol.append(fk.toREEString()).append(" ^ ");
                    }
                }
            }
        }

        return tables.toString() + joinCol + reeStr;
    }

    private Map<String, PredicateSetTableMsg> generateFkMsg(Map<String, PredicateSetTableMsg> colMap) {
        if (ONLY_ONE < colMap.keySet().size() && null != FKMap) {
            String [] tableNames = colMap.keySet().toArray(new String []{});
            for(int i = 0; i < tableNames.length; i++) {
                for(int j = i+1; j < tableNames.length; j++) {
                    String useTables = tableNames[i] + MlsConstant.RD_JOINTABLE_SPLIT_STR + tableNames[j];
                    if(!FKMap.containsKey(useTables)) {
                        useTables = tableNames[j] + MlsConstant.RD_JOINTABLE_SPLIT_STR + tableNames[i];
                        if(!FKMap.containsKey(useTables)) {
                            continue;
                        }
                    }
                    addFkIntoTable(FKMap.get(useTables), colMap, useTables);
                }
            }
        }
        return colMap;
    }

    private void addFkIntoTable(Set<Predicate> fkSet, Map<String, PredicateSetTableMsg> colMap, String useTables) {
        for (Predicate fk : fkSet) {
            String [] joinTables = useTables.split(MlsConstant.RD_JOINTABLE_SPLIT_STR);

            Integer alias = colMap.get(joinTables[0]).getTableAlias().iterator().next();
            fk.setIndexOperand1(alias);
            alias = colMap.get(joinTables[1]).getTableAlias().iterator().next();
            fk.setIndexOperand2(alias);

            colMap.get(joinTables[0]).addJoinSet(fk);
            String [] array = fk.getOperand1().toString_(fk.getIndex1()).split("[.]");
            colMap.get(joinTables[0]).addColName(array[2].substring(array[2].lastIndexOf("__")+2));

            array = fk.getOperand2().toString_(fk.getIndex2()).split("[.]");
            colMap.get(joinTables[1]).addColName(array[2].substring(array[2].lastIndexOf("__")+2));
        }
    }

    @Override
    public int hashCode() {
        return (currentList.toString() + "->" + this.rhs.toString()).hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        REE other = (REE) obj;
        return other.currentList.equals(this.currentList) && other.rhs.equals(this.rhs);
    }

}
