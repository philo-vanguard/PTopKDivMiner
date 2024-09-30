package sics.seiois.mlsserver.service.impl;

import com.alibaba.fastjson.JSONObject;
import com.sics.seiois.client.dto.request.mls.RuleDiscoverExecuteRequest;
import com.sics.seiois.client.model.mls.ColumnInfo;
import com.sics.seiois.client.model.mls.JoinInfo;
import com.sics.seiois.client.model.mls.JoinInfos;
import com.sics.seiois.client.model.mls.ModelPredicate;
import com.sics.seiois.client.model.mls.TableInfo;
import com.sics.seiois.client.model.mls.TableInfos;

import de.metanome.algorithm_integration.Operator;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

import sics.seiois.mlsserver.biz.ConstantPredicateGenerateMain;
import sics.seiois.mlsserver.biz.der.metanome.REE;
import sics.seiois.mlsserver.biz.der.metanome.REEFinderEvidSet;
import sics.seiois.mlsserver.biz.der.metanome.denialconstraints.DenialConstraint;
import sics.seiois.mlsserver.biz.der.metanome.denialconstraints.DenialConstraintSet;
import sics.seiois.mlsserver.biz.der.metanome.pojo.RuleResult;
import sics.seiois.mlsserver.biz.der.metanome.predicates.Predicate;
import sics.seiois.mlsserver.biz.der.metanome.predicates.PredicateBuilder;
import sics.seiois.mlsserver.biz.der.metanome.predicates.sets.PredicateSet;
import sics.seiois.mlsserver.biz.der.mining.Evaluation;
import sics.seiois.mlsserver.biz.der.mining.ParallelRuleDiscoveryTopkDiversified;
import sics.seiois.mlsserver.enums.RuleFinderTypeEnum;
import sics.seiois.mlsserver.model.*;
import sics.seiois.mlsserver.utils.RuntimeParamUtil;
import sics.seiois.mlsserver.utils.helper.ErRuleFinderHelper;

public class EvidenceGenerateMain {

    private static final Logger logger = LoggerFactory.getLogger(EvidenceGenerateMain.class);
    public final static String ORC_TYPE_SUFFIX = "orc";
    public final static String CSV_TYPE_SUFFIX = "csv";
    public static final String MLS_TMP_HOME = "/tmp/rulefind/";
    private static final String CONFIG_PATH = "predict.conf";
    private static final StringBuffer timeInfo = new StringBuffer();
    private static final int NORECORD = 0;


    public static void main(String[] args) throws Exception {
        // 本地文件 /tmp/rulefind/{taskId}/{taskId}_req.json
        String requestJsonFileName = args[0];
        logger.info("#### rulefinder request param file name:{}", requestJsonFileName);

        SparkSession spark = SparkSession.builder().appName("evidenceset").getOrCreate();
        File tableJsonFile = new File(requestJsonFileName);
        String reqJson = FileUtils.readFileToString(tableJsonFile, "utf-8");
        logger.info("#### reqJson:{}", reqJson);

        RuleDiscoverExecuteRequest request = JSONObject.parseObject(reqJson).toJavaObject(RuleDiscoverExecuteRequest.class);
        String taskId = request.getTaskId();
        String runtimeParam = request.getOtherParam();
        TableInfos tableInfos = request.getTableInfos();

        Set<String> usefulCol = new HashSet<>();
        for(TableInfo tableInfo : tableInfos.getTableInfoList()) {
            for (ColumnInfo columnInfo : tableInfo.getColumnList()) {
                if(!columnInfo.getSkip()) {
                    usefulCol.add(columnInfo.getColumnName());
                }
            }
        }
        if(usefulCol.size() <= NORECORD) {
            spark.close();
            return;
        }
        spark.conf().set("taskId", taskId);
        spark.conf().set("runtimeParam", runtimeParam);

        logger.info(">>>>show run time params: {}", runtimeParam);
        PredicateConfig.getInstance(CONFIG_PATH).setRuntimeParam(runtimeParam);

        FileSystem hdfs = FileSystem.get(new Configuration());
        //对该taskId的临时目录进行清理
        if (hdfs.exists(new Path(MLS_TMP_HOME + taskId))) {
            hdfs.delete(new Path(MLS_TMP_HOME + taskId), true);
        }

        //根据配置文件是否需要走分布式方案
        PredicateConfig config = PredicateConfig.getInstance(CONFIG_PATH);
        JoinInfos joinInfos = null;

        joinInfos = DataDealerOnSpark.getJoinTablesOnly(tableInfos, taskId);
        generateESAndRuleDig(request, joinInfos, spark, config);

        spark.close();
    }

    /**
     * 生成ES并且进行分布式规则挖掘
     * @throws Exception
     */
    private static void generateESAndRuleDig(RuleDiscoverExecuteRequest request, JoinInfos joinInfos,
                                             SparkSession spark,
                                             PredicateConfig config) throws Exception {

        logger.info("#### generateESAndRuleDig start.joinInfos.size={}", joinInfos.getJoinInfoList().size());
        FileSystem hdfs = FileSystem.get(new Configuration());
        String taskId = request.getTaskId();
        TableInfos tableInfos = request.getTableInfos();
        List<ModelPredicate> modelPredicateList = request.getModelPredicats();

        //join表转成map
        Map<String, List<JoinInfo>> tblNum2List = new HashMap<>();
        for (JoinInfo joinTable : joinInfos.getJoinInfoList()) {
            int num = joinTable.getTableNumber();
            tblNum2List.computeIfAbsent(String.valueOf(num), k -> new ArrayList<>()).add(joinTable);
        }

        //开始清空规则输出目录，对上次执行输出的结果清空
        Path outputPath = new Path(request.getResultStorePath());
        hdfs.delete(outputPath, true);

        //第一步1-单表数据准备。tableInfos 里面如果是orc格式的表转成csv格式的表，并且在spark环境创建了虚拟表
        prepareData(taskId, tableInfos, hdfs, spark);

        /************* CR ER 规则发现主入口***************/
        List<RuleResult> finalRules = new ArrayList<>();
        if(RuleFinderTypeEnum.ER_ONLY.getValue().equals(request.getType())) {
            finalRules = generateErRule(request, joinInfos, spark, hdfs, config);
        } else if(RuleFinderTypeEnum.CR_ONLY.getValue().equals(request.getType())) {
            finalRules = generateCrRule(request, joinInfos, spark, hdfs, config);
        } else {
            //如果ER规则发现参数不为空，进行ER规则发现。ER有异常也继续往下走不影响主流程
            if(request.getErRuleConfig() != null) {
                try {
                    finalRules = generateErRule(request, joinInfos, spark, hdfs, config);
                } catch (Exception e) {
                    logger.error("#### erRuleFinder exception", e);
                }
            }
            //进行CR规则发现
            List<RuleResult> crRules = generateCrRule(request, joinInfos, spark, hdfs, config);
            finalRules.addAll(crRules);
        }

        logger.info("####[规则发现完成,数量:{}, 保存到HDFS开始]", finalRules.size());
        writeRuleToHDFS(hdfs, finalRules, PredicateConfig.MLS_TMP_HOME + taskId + "/allrule.txt");
//        writeRuleToHDFS(request, hdfs, finalRules);
        logger.info("####[规则发现完成,数量:{}, 保存到HDFS结束]", finalRules.size());
    }

    private static List<RuleResult> generateErRule(RuleDiscoverExecuteRequest request, JoinInfos joinInfos, SparkSession spark, FileSystem hdfs,
                                                   PredicateConfig config) throws Exception {
        logger.info("#### erRuleFinder start!tableInfos={}", request.getTableInfos().toString());
        String eidName = ErRuleFinderHelper.EID_NAME_PREFIX + request.getErRuleConfig().getEidName();
        request.getErRuleConfig().setEidName(eidName);

        //根据标注数据获取新的tableInfos
        TableInfos markedTableInfos = ErRuleFinderHelper.getMarkedTableInfos(spark, hdfs, request);
        logger.info("#### erRuleFinder gen tableInfos success! new tableInfos={}", markedTableInfos.toString());

        /***************ER规则发现结束*************************/
        //进行规则发现
        return generateRule(request, markedTableInfos, joinInfos, hdfs, spark, config, true, eidName);
    }

    private static List<RuleResult> generateCrRule(RuleDiscoverExecuteRequest request, JoinInfos joinInfos, SparkSession spark, FileSystem hdfs,
                                                   PredicateConfig config) throws Exception {
        logger.info("#### crRuleFinder start!tableInfos={}", request.getTableInfos().toString());
        TableInfos tableInfos = request.getTableInfos();

        return generateRule(request, tableInfos, joinInfos, hdfs, spark, config, false, null);
    }

    private static void prepareData(String taskId, TableInfos tableInfos, FileSystem hdfs, SparkSession spark) throws IOException {
        boolean deleteTmpData = false;
        for (TableInfo tableInfo : tableInfos.getTableInfoList()) {
            String tablePath = tableInfo.getTableDataPath();
            String newTablePath = PredicateConfig.MLS_TMP_HOME + taskId + "/" + tableInfo.getTableName() + "/" + tableInfo.getTableName() + ".csv";

            if (!tablePath.endsWith(".csv")) {
                spark.read().format("orc").load(tablePath).write().format("csv").option("header", "true").save(newTablePath);
                tableInfo.setTableDataPath(newTablePath);
                deleteTmpData = false;
            }
        }
        for (int i = 0; i < tableInfos.getTableInfoList().size(); i++) {
            String tablename = tableInfos.getTableInfoList().get(i).getTableName();
            String tableLocation = tableInfos.getTableInfoList().get(i).getTableDataPath();
            if (tableLocation.endsWith(CSV_TYPE_SUFFIX)) {
                //创建表
                logger.info("****a:");
                spark.read().format("com.databricks.spark.csv").option("header", "true").load(tableLocation).createOrReplaceTempView(tablename);
            } else {
                logger.info("****b:");
                spark.read().format("orc").load(tableLocation).createOrReplaceTempView(tablename);
            }
            logger.info("****表格" + tablename);
        }

        if (deleteTmpData) {
            for (TableInfo tableInfo : tableInfos.getTableInfoList()) {
                hdfs.delete(new Path(tableInfo.getTableDataPath()), true);
            }
        }
    }

    private static List<RuleResult> generateRule(RuleDiscoverExecuteRequest request, TableInfos tableInfos, JoinInfos joinInfos, FileSystem hdfs,
                                     SparkSession spark, PredicateConfig config, boolean erRuleFinderFlag, String eidName) throws Exception {
        String taskId = request.getTaskId();
        long startTime = System.currentTimeMillis();
        List<String> constantPredicate = generateConstant(taskId, tableInfos, spark, config, erRuleFinderFlag, eidName);
        logger.info("#### constantPredicate size: {}", constantPredicate.size());
        logger.info("#### constantPredicate: {}", constantPredicate);

        // randomly select 5 constant predicates to participate in rule mining
//        List<String> selected_constantPredicate = new ArrayList<>();
//        selectConstantPredicates(constantPredicate, selected_constantPredicate, 5);
//        logger.info("#### selected_constantPredicate size: {}", selected_constantPredicate.size());
//        logger.info("#### selected_constantPredicate: {}", selected_constantPredicate);

        SparkContextConfig sparkContextConfig = new SparkContextConfig(config, request, tableInfos);

        String algOption = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "algOption");
        logger.info("####predicateConfig:{}", config.toString());
        //(1)索引方式生成ES关键逻辑
        REEFinderEvidSet reeFinderEvidSet = new REEFinderEvidSet();
        // set alg option
        reeFinderEvidSet.setAlgOption(algOption);
        List<RuleResult> allRules = new ArrayList<>();
        spark.conf().set("RuleCount", 0 + "->" + allRules.size());

        reeFinderEvidSet.generateInput(taskId, tableInfos, joinInfos, spark, constantPredicate, config, eidName);

        // write experimental results
        long inputTime = System.currentTimeMillis() - startTime;
        timeInfo.append("input data time: ").append(inputTime).append("\n");

        if(!config.isExecuteOnlyConstantPred()){
            logger.info("####predicate_type!=3，生成多行规则");
            if (algOption.equals("diversified")) {
                allRules.addAll(generateMultiTupleRuleDiversified(request, reeFinderEvidSet, tableInfos, hdfs, spark, config));
            } else if (algOption.equals("evaluation")) {
                evaluateREEs(request, reeFinderEvidSet, tableInfos, hdfs, spark, config);
            } else if (algOption.equals("obtainSuboptimalResults")) {
                obtainSuboptimalResults(request, reeFinderEvidSet, tableInfos, hdfs, spark, config);
            }
        }

        String outputResultFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "outputResultFile");

        //打印规则
//        for(RuleResult rule : allRules) {
//            logger.info("####[输出规则]:" + rule.getPrintString());
//        }

        long runningTime = System.currentTimeMillis() - startTime;
        timeInfo.append("total running time: ").append(runningTime).append("\n");

        String outTxtPath = PredicateConfig.MLS_TMP_HOME + taskId + "/rule_all/" + outputResultFile; // "experiment_results";
        FSDataOutputStream outTxt = hdfs.create(new Path(outTxtPath));
        outTxt.writeBytes(timeInfo.toString());
        outTxt.close();

        return allRules;
    }


    public static DenialConstraint parseOneREE(String rule, REEFinderEvidSet reeFinderEvidSet) {
        String[] pstrings = rule.split(",")[0].split("->");
        String[] xStrings = pstrings[0].split("\\^");
        String[] xString = xStrings[xStrings.length - 1].trim().split("  ");
        PredicateSet ps = new PredicateSet();
        for (int i = 0; i < xString.length; i++) {
            ps.add(PredicateBuilder.parsePredicateString(reeFinderEvidSet.getInput(), xString[i].trim()));
        }
        String rhs = pstrings[pstrings.length - 1].trim();
        ps.addRHS(PredicateBuilder.parsePredicateString(reeFinderEvidSet.getInput(), rhs.substring(0, rhs.length() - 1)));
        DenialConstraint ree = new DenialConstraint(ps);
        ree.removeRHS();
        return ree;
    }

    private static DenialConstraintSet loadREEs(String taskId, String outputResultFile, FileSystem hdfs, REEFinderEvidSet reeFinderEvidSet) throws Exception {
        DenialConstraintSet rees = new DenialConstraintSet();
        String inputTxtPath = PredicateConfig.MLS_TMP_HOME + "constantRecovery/" + taskId + "/rule_all/" +  outputResultFile; //"experiment_results";
        FSDataInputStream inputTxt = hdfs.open(new Path(inputTxtPath));
        BufferedInputStream bis = new BufferedInputStream(inputTxt);
        InputStreamReader sReader = new InputStreamReader(bis, "UTF-8");
        BufferedReader bReader = new BufferedReader(sReader);
        String line;
        String valid_prefix = "Rule : ";
        while ((line = bReader.readLine()) != null) {
            if (line.length() <= valid_prefix.length() || (!line.startsWith(valid_prefix))) {
                continue;
            } else {
                DenialConstraint ree = parseOneREE(line.trim(), reeFinderEvidSet);
                if (ree != null) {
                    rees.add(ree);
                }
            }
        }
        logger.info("#### load {} rees", rees.size());
        for (DenialConstraint ree : rees) {
            logger.info(ree.toString());
        }
        return rees;
    }

    /*
        for top-k diversified algorithm
     */
    private static List<RuleResult> generateMultiTupleRuleDiversified(RuleDiscoverExecuteRequest request, REEFinderEvidSet reeFinderEvidSet,
                                                           TableInfos tableInfos, FileSystem hdfs, SparkSession spark,
                                                           PredicateConfig config) throws Exception {

        logger.info("#### generateMultiTupleRuleDiversified start----------------------------------------------------{}", spark.conf().get("runtimeParam"));
        SparkContextConfig sparkContextConfig = new SparkContextConfig(config, request, tableInfos);
        // collect all predicates, including ML, constant and other traditional predicates.
        // for paper: do not consider comparison predicates, i.e., <, >, <=, >=
        List<Predicate> allPredicates = new ArrayList<>();
        int ccount = 0;
        for (Predicate p : reeFinderEvidSet.getPredicateBuilder().getPredicates()) {
            if (p.isML() || p.getOperator() == Operator.EQUAL) {
                allPredicates.add(p);
            }
            ccount++;
        }
        logger.info("#### non-constant predicates size: {}", allPredicates.size());
        for (Predicate p : reeFinderEvidSet.getConstantPredicateBuilder().getPredicates()) {
            if (p.getOperator() == Operator.EQUAL) {
                allPredicates.add(p);
            }
        }
        logger.info("#### allPredicates size: {}", allPredicates.size());

        logger.info("#### allPredicates: {}", allPredicates);

//        writeAllPredicatesToFile(allPredicates, reeFinderEvidSet.getInput().getName());

        int maxOneRelationNum = reeFinderEvidSet.getInput().getMaxTupleOneRelation();
        int allCount = reeFinderEvidSet.getInput().getAllCount();
        int maxTupleVariableNum = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"),"maxTupleVariableNum"));
        int K = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "topK"));
        long support = (long)(sparkContextConfig.getCr() * allCount * allCount);
        double confidence = sparkContextConfig.getFtr();
        logger.info("support threshold: {}, confidence threshold: {}, allCount: {}", support, confidence, allCount);
        boolean ifPrune = Boolean.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "ifPrune"));

        int if_cluster_workunits = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "ifClusterWorkunits"));

        int filter_enum_number = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "filterEnumNumber"));

        String outputResultFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "outputResultFile");

        String relevance = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "relevance");
        String diversity = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "diversity");
        double lambda = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "lambda"));
        double sampleRatioForTupleCov = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "sampleRatioForTupleCov"));

        double w_supp = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_supp"));
        double w_conf = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_conf"));
        double w_rel_model = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_rel_model"));
        double w_attr_non = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_attr_non"));
        double w_pred_non = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_pred_non"));
        double w_attr_dis = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_attr_dis"));
        double w_tuple_cov = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_tuple_cov"));

        String predicateToIDFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "predicateToIDFile");
        String relevanceModelFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "relevanceModelFile");

        int MAX_X_LENGTH = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "MAX_X_LENGTH"));

        ParallelRuleDiscoveryTopkDiversified parallelRuleDiscovery = new ParallelRuleDiscoveryTopkDiversified(allPredicates, maxTupleVariableNum,
                support, (float)confidence, maxOneRelationNum, reeFinderEvidSet.getInput(), allCount,
                ifPrune, if_cluster_workunits, filter_enum_number,
                K, relevance, diversity, lambda, sampleRatioForTupleCov,
                w_supp, w_conf, w_rel_model, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov,
                predicateToIDFile, relevanceModelFile, hdfs,
                reeFinderEvidSet.getIndex_null_string(), reeFinderEvidSet.getIndex_null_double(), reeFinderEvidSet.getIndex_null_long(),
                MAX_X_LENGTH);

        String taskId = request.getTaskId();

        long startMineTime = System.currentTimeMillis();
        if (diversity.equals("null")) {  // top-k
            parallelRuleDiscovery.levelwiseRuleDiscoveryTopK(taskId, spark, 0);
        } else {                         // top-k diversified
            // the first iteration
            int iteration = 0;
            logger.info("#### iteration: {}", iteration);
            boolean ifContinue;
            long start = System.currentTimeMillis();
            ifContinue = parallelRuleDiscovery.levelwiseRuleDiscoveryFirstIteration(taskId, spark, iteration);
            logger.info("#### In iteration {}, REE Discovery Time: {}", iteration, (System.currentTimeMillis() - start) / 1000);
            iteration++;
            // the rest iteration
            while (ifContinue && iteration < K) {
                logger.info("#### iteration: {}", iteration);
                start = System.currentTimeMillis();
                ifContinue = parallelRuleDiscovery.levelwiseRuleDiscovery(taskId, spark, iteration);
                logger.info("#### In iteration {}, REE Discovery Time: {}", iteration, (System.currentTimeMillis() - start) / 1000);
                iteration++;
            }
        }
        long duringTime = System.currentTimeMillis() - startMineTime;

        timeInfo.append("REE Discovery Time : ").append(duringTime).append("\n");
        logger.info("#### REE Discovery Time: {}", duringTime);

        // save rules
        ArrayList<REE> rees = parallelRuleDiscovery.getFinalResults();
        for (REE ree : rees) {
            if (ree == null) {
                continue;
            }
//            timeInfo.append("Rule : ").append(ree.toREEString()).append(", supp: ").append(ree.getSupport()).append(", conf:").append(ree.getConfidence())
//                    .append(", rel_score:").append(ree.getRelevance_score()).append(", div_score:").append(ree.getDiversity_score()).append(", score:").append(ree.getScore())
//                    .append(", supp_score:").append(ree.getSupp_score()).append(", attr_nonoverlap:").append(ree.getAttr_nonoverlap_score())
//                    .append(", pred_nonoverlap:").append(ree.getPred_nonoverlap_score()).append(", tuple_cov:").append(ree.getTuple_cov_score()).append("\n");

            timeInfo.append("Rule : ").append(ree.toString()).append(", supp: ").append(ree.getSupport()).append(", conf:").append(ree.getConfidence())
                    .append(", rel_score:").append(ree.getRelevance_score()).append(", div_score:").append(ree.getDiversity_score()).append(", score:").append(ree.getScore()).append("\n");
        }

        String outTxtPath = PredicateConfig.MLS_TMP_HOME + taskId + "/rule_all/" +  outputResultFile; //"experiment_results";
        FSDataOutputStream outTxt = hdfs.create(new Path(outTxtPath));
        outTxt.writeBytes(timeInfo.toString());
        outTxt.close();

//        BufferedOutputStream bos = new BufferedOutputStream(outTxt);
//        OutputStreamWriter sWriter = new OutputStreamWriter(bos, "UTF-8");
//        BufferedWriter bWriter = new BufferedWriter(sWriter);
//        bWriter.write(timeInfo.toString());
//        outTxt.close();

        logger.info("#### REEs display----------------------------------------------------");
        int topk = 1;
        for (REE ree : rees) {
            if (ree == null) {
                continue;
            }
            logger.info("TOP {}. {}", topk, ree.toString());
            topk ++;
        }

        List<RuleResult> ruleResults = new ArrayList<>();

        // final print
        int reeIndex = 0;
        for (REE ree : rees) {
            if (ree == null) {
                continue;
            }
            reeIndex++;
            RuleResult rule = new RuleResult();
            rule.setID(reeIndex);
            rule.setSupport(ree.getSupport());
            rule.setUnsupport(ree.getViolations());
            rule.setRule(ree.toString());

            ruleResults.add(rule);
        }

        return ruleResults;
    }


    /*
        for evaluation
     */
    private static void evaluateREEs(RuleDiscoverExecuteRequest request, REEFinderEvidSet reeFinderEvidSet,
                                 TableInfos tableInfos, FileSystem hdfs, SparkSession spark,
                                 PredicateConfig config) throws Exception {

        logger.info("#### evaluation start----------------------------------------------------{}", spark.conf().get("runtimeParam"));
        SparkContextConfig sparkContextConfig = new SparkContextConfig(config, request, tableInfos);
        ArrayList<Predicate> allPredicates = new ArrayList<>();
        for (Predicate p : reeFinderEvidSet.getPredicateBuilder().getPredicates()) {
            if (p.isML() || p.getOperator() == Operator.EQUAL) {
                allPredicates.add(p);
            }
        }
        logger.info("#### non-constant predicates size: {}", allPredicates.size());
        for (Predicate p : reeFinderEvidSet.getConstantPredicateBuilder().getPredicates()) {
            if (p.getOperator() == Operator.EQUAL) {
                allPredicates.add(p);
            }
        }
        logger.info("#### allPredicates size: {}", allPredicates.size());

        logger.info("#### allPredicates: {}", allPredicates);

        int maxOneRelationNum = reeFinderEvidSet.getInput().getMaxTupleOneRelation();
        int allCount = reeFinderEvidSet.getInput().getAllCount();
        String taskId = request.getTaskId();

        int maxTupleVariableNum = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"),"maxTupleVariableNum"));
        double sampleRatioForTupleCov = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "sampleRatioForTupleCov"));
        double lambda = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "lambda"));

        double w_supp = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_supp"));
        double w_conf = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_conf"));
        double w_rel_model = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_rel_model"));
        double w_attr_non = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_attr_non"));
        double w_pred_non = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_pred_non"));
        double w_attr_dis = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_attr_dis"));
        double w_tuple_cov = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_tuple_cov"));

        String predicateToIDFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "predicateToIDFile");
        String relevanceModelFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "relevanceModelFile");

        String inputREEsPath = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "inputREEsPath");
        String outputResultFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "outputResultFile");


        Evaluation evaluation = new Evaluation(allPredicates, inputREEsPath, maxTupleVariableNum,
                taskId, maxOneRelationNum, allCount, sampleRatioForTupleCov, lambda,
                w_supp, w_conf, w_rel_model, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov,
                predicateToIDFile, relevanceModelFile, hdfs,
                reeFinderEvidSet.getIndex_null_string(), reeFinderEvidSet.getIndex_null_double(), reeFinderEvidSet.getIndex_null_long(),
                true);

        evaluation.loadREEs();

        evaluation.printREEs();

        long startMineTime = System.currentTimeMillis();
        evaluation.evaluate(spark, timeInfo);
        long duringTime = System.currentTimeMillis() - startMineTime;

        timeInfo.append("The Evaluation Time:").append(duringTime).append("\n");

        String outTxtPath = PredicateConfig.MLS_TMP_HOME + taskId + "/rule_all/" +  outputResultFile; //"experiment_results";
        FSDataOutputStream outTxt = hdfs.create(new Path(outTxtPath));
        outTxt.writeBytes(timeInfo.toString());
        outTxt.close();
    }


    /*
    * use swapping algorithm to obtain suboptimal top-k diversified results
    * */
    public static void obtainSuboptimalResults(RuleDiscoverExecuteRequest request, REEFinderEvidSet reeFinderEvidSet,
                                               TableInfos tableInfos, FileSystem hdfs, SparkSession spark,
                                               PredicateConfig config) throws Exception {
        logger.info("#### obtainSuboptimalResults start----------------------------------------------------{}", spark.conf().get("runtimeParam"));
        SparkContextConfig sparkContextConfig = new SparkContextConfig(config, request, tableInfos);
        ArrayList<Predicate> allPredicates = new ArrayList<>();
        for (Predicate p : reeFinderEvidSet.getPredicateBuilder().getPredicates()) {
            if (p.isML() || p.getOperator() == Operator.EQUAL) {
                allPredicates.add(p);
            }
        }
        logger.info("#### non-constant predicates size: {}", allPredicates.size());
        for (Predicate p : reeFinderEvidSet.getConstantPredicateBuilder().getPredicates()) {
            if (p.getOperator() == Operator.EQUAL) {
                allPredicates.add(p);
            }
        }
        logger.info("#### allPredicates size: {}", allPredicates.size());

        logger.info("#### allPredicates: {}", allPredicates);

        int maxOneRelationNum = reeFinderEvidSet.getInput().getMaxTupleOneRelation();
        int allCount = reeFinderEvidSet.getInput().getAllCount();
        String taskId = request.getTaskId();

        int maxTupleVariableNum = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"),"maxTupleVariableNum"));
        double sampleRatioForTupleCov = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "sampleRatioForTupleCov"));
        double lambda = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "lambda"));

        double w_supp = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_supp"));
        double w_conf = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_conf"));
        double w_rel_model = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_rel_model"));
        double w_attr_non = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_attr_non"));
        double w_pred_non = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_pred_non"));
        double w_attr_dis = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_attr_dis"));
        double w_tuple_cov = Double.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "w_tuple_cov"));

        String predicateToIDFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "predicateToIDFile");
        String relevanceModelFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "relevanceModelFile");

        String inputREEsPath = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "inputREEsPath");
        String outputResultFile = RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "outputResultFile");

        int topKNum = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "topK"));

        int iter_subopt = Integer.valueOf(RuntimeParamUtil.getRuntimeParam(spark.conf().get("runtimeParam"), "iter_subopt"));

        /*
        * Run once!
        * */
        /*
        // 1. ----------------------------------- for obtaining suboptimal results -----------------------------------
        Evaluation obtainSuboptimal = new Evaluation(allPredicates, inputREEsPath, maxTupleVariableNum,
                taskId, maxOneRelationNum, allCount, sampleRatioForTupleCov, lambda,
                w_supp, w_conf, w_rel_model, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov,
                predicateToIDFile, relevanceModelFile, hdfs,
                reeFinderEvidSet.getIndex_null_string(), reeFinderEvidSet.getIndex_null_double(), reeFinderEvidSet.getIndex_null_long(),
                topKNum);

        long startMineTime = System.currentTimeMillis();
        ArrayList<REE> selected = obtainSuboptimal.swapping(spark);
        long duringTime = System.currentTimeMillis() - startMineTime;

        logger.info("The obtainSuboptimalResults Time: {}", duringTime);
        timeInfo.append("The obtainSuboptimalResults Time:").append(duringTime).append("\n");


        // 2. ------------------------------------------- for evaluation -------------------------------------------
        Evaluation evaluation = new Evaluation(allPredicates, inputREEsPath, maxTupleVariableNum,
                taskId, maxOneRelationNum, allCount, sampleRatioForTupleCov, lambda,
                w_supp, w_conf, w_rel_model, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov,
                predicateToIDFile, relevanceModelFile, hdfs,
                reeFinderEvidSet.getIndex_null_string(), reeFinderEvidSet.getIndex_null_double(), reeFinderEvidSet.getIndex_null_long(),
                false);

        evaluation.loadREEs(selected);

        evaluation.printREEs();

        startMineTime = System.currentTimeMillis();
        evaluation.evaluate(spark, timeInfo);
        duringTime = System.currentTimeMillis() - startMineTime;

        logger.info("The evaluation Time: {}", duringTime);
        timeInfo.append("The evaluation Time:").append(duringTime).append("\n");

        String outTxtPath = PredicateConfig.MLS_TMP_HOME + taskId + "/rule_all/" +  outputResultFile; //"experiment_results";
        FSDataOutputStream outTxt = hdfs.create(new Path(outTxtPath));
        outTxt.writeBytes(timeInfo.toString());
        outTxt.close();
         */


        /*
        * Run multiple times!
        * */
        // 1. ----------------------------------- for obtaining suboptimal results -----------------------------------
        Evaluation obtainSuboptimal = new Evaluation(allPredicates, inputREEsPath, maxTupleVariableNum,
                taskId, maxOneRelationNum, allCount, sampleRatioForTupleCov, lambda,
                w_supp, w_conf, w_rel_model, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov,
                predicateToIDFile, relevanceModelFile, hdfs,
                reeFinderEvidSet.getIndex_null_string(), reeFinderEvidSet.getIndex_null_double(), reeFinderEvidSet.getIndex_null_long(),
                topKNum);

//        // 2. ------------------------------------------- for evaluation -------------------------------------------
//        Evaluation evaluation = new Evaluation(allPredicates, inputREEsPath, maxTupleVariableNum,
//                taskId, maxOneRelationNum, allCount, sampleRatioForTupleCov, lambda,
//                w_supp, w_conf, w_rel_model, w_attr_non, w_pred_non, w_attr_dis, w_tuple_cov,
//                predicateToIDFile, relevanceModelFile, hdfs,
//                reeFinderEvidSet.getIndex_null_string(), reeFinderEvidSet.getIndex_null_double(), reeFinderEvidSet.getIndex_null_long(),
//                false);

        // ---------------------------------------------------------------------------------------------------------
        for (int iter = 0; iter < iter_subopt; iter++) {
            // ----------------------------------------------------------------------------------------------------
            logger.info("--------------------- iter {} ---------------------", iter);
            timeInfo.append("\n\n--------------------- iter ").append(iter).append(" ---------------------").append("\n");
            long startMineTime = System.currentTimeMillis();
            ArrayList<REE> selected = obtainSuboptimal.swapping(spark);
            long duringTime = System.currentTimeMillis() - startMineTime;

            logger.info("The obtainSuboptimalResults Time: {}", duringTime);
            timeInfo.append("The obtainSuboptimalResults Time:").append(duringTime).append("\n");

            timeInfo.append("#### Selected top-k diversified REEs:\n");
            for (REE ree : selected) {
                timeInfo.append(ree.toString()).append("\n");
            }

            // --------------------------------- results computed below are NOT right --------------------------------
//            evaluation.loadREEs(selected);
//
//            evaluation.printREEs();
//
//            startMineTime = System.currentTimeMillis();
//            evaluation.evaluate(spark, timeInfo);
//            duringTime = System.currentTimeMillis() - startMineTime;
//
//            logger.info("The evaluation Time: {}", duringTime);
//            timeInfo.append("The evaluation Time:").append(duringTime).append("\n");

        }

        String outTxtPath = PredicateConfig.MLS_TMP_HOME + taskId + "/rule_all/" +  outputResultFile; //"experiment_results";
        FSDataOutputStream outTxt = hdfs.create(new Path(outTxtPath));
        outTxt.writeBytes(timeInfo.toString());
        outTxt.close();
    }



    public static List<RuleResult> getAllRule(RuleDiscoverExecuteRequest request, List<RuleResult> allRuleRlt) {
        String taskId = request.getTaskId();
        String outRulePath = PredicateConfig.MLS_TMP_HOME + taskId + "/allrule.txt";

        logger.info("######ALL RULE PATH: " + outRulePath);
        long startMinCoverTime = System.currentTimeMillis();
        List<RuleResult> finalRuleRlt = new ArrayList<>();
        for(RuleResult rule : allRuleRlt) {
            finalRuleRlt.add(rule);
        }

        long endMinCoverTime = System.currentTimeMillis();

        long mincoverTime = endMinCoverTime - startMinCoverTime;
        timeInfo.append("mincover time: ").append(mincoverTime).append("\n");

        return finalRuleRlt;
    }


    private static List<String> generateConstant(String taskId, TableInfos tableInfos, SparkSession spark, PredicateConfig config,
                                                 boolean erRuleFinderFlag, String eidName) {
        if (!config.isExecuteOnlyNonConstantPred()) {
            List<String> allTablePredicates = new ArrayList<>();
            for (TableInfo tableInfo : tableInfos.getTableInfoList()) {
                List<String> tablePredicate = null;
                List<Row> constantRow = ConstantPredicateGenerateMain.geneConstantList(spark, tableInfo, config, erRuleFinderFlag, eidName);

                logger.info("####{},{} generate constant row {} 条", taskId, tableInfo.getTableName(), constantRow.size());
                tablePredicate = ConstantPredicateGenerateMain.genePredicateList(constantRow, tableInfo, config);
                logger.info("####{},{} generate constant predicate {} 条", taskId, tableInfo.getTableName(), tablePredicate.size());
                logger.info("####{},{} generate constant predicate {}", taskId, tableInfo.getTableName(), tablePredicate.toString());
                allTablePredicates.addAll(tablePredicate);
            }
            return allTablePredicates;
        }
        return new ArrayList<>();
    }

    public static void writeAllPredicatesToFile(List<Predicate> allPredicates, String table_name) {
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter("/opt/allPredicates_" + table_name + ".txt"));
            for (Predicate p : allPredicates) {
                out.write(p.toString());
                out.write("\n");
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeRuleToHDFS(FileSystem hdfs, List<RuleResult> finalRR, String path) throws IOException {
//        logger.info("######PATH: " + path);
        FSDataOutputStream outTxt = hdfs.create(new Path(path));
        for (RuleResult rule : finalRR) {
            //备份的结果只要保留REE格式即可
            outTxt.write(rule.getPrintREEs().getBytes("UTF-8"));
            outTxt.writeBytes("\n");
        }
        outTxt.close();
    }
}
