package sics.seiois.mlsserver.biz.der.mining.utils;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sics.seiois.mlsserver.biz.der.metanome.REE;

import java.io.Serializable;
import java.util.ArrayList;


public class Relevance implements Serializable {
    private static final long serialVersionUID = 1624566984357674100L;
    private static Logger logger = LoggerFactory.getLogger(Relevance.class);

    private ArrayList<String> candidate_functions;

    public Relevance() {
        this.candidate_functions = new ArrayList<>();
        this.candidate_functions.add("support");
        this.candidate_functions.add("confidence");
        this.candidate_functions.add("relevance_model");
        this.candidate_functions.add("template");
        this.candidate_functions.add("unexpectedness");
    }

    public Relevance(String relevance) {
        this.candidate_functions = new ArrayList<>();
        if (relevance.equals("null")) {
            return;
        }
        for (String func : relevance.split("#")) {
            this.candidate_functions.add(func);
        }
        logger.info("#### relevance measures: {}", this.candidate_functions);
    }

    public ArrayList<String> getCandidate_functions() {
        return this.candidate_functions;
    }


    public double computeUB(REE ree, double w_supp, double w_conf,
                            RelevanceModel relevance_model, double w_rel_model) {
        double UB = 0.0;
        for (String rel : this.candidate_functions) {
            switch (rel) {
                case "support":
                    double UB_supp = ree.getSupp_score();
                    UB += w_supp * UB_supp;
                    break;
                case "confidence":
                    double UB_conf = 1.0;
                    UB += w_conf * UB_conf;
                    break;
                case "relevance_model":
                    double UB_rel_model = relevance_model.getWeightUBSubj();
                    UB += w_rel_model * UB_rel_model;
                    break;
            }
        }
        return UB;
    }
}
