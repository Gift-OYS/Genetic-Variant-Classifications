# Genetic-Variant-Classifications
# 一、项目背景
“21世纪是生物的世纪”，20世纪70年代后，生物科学的新进展如雨后春笋，层出不穷。当代生物科学主要朝着微观和宏观两个方向进展，而从微观角度，遗传变异、基因突变等领域是一个非常有前景和生命力的方向，而基因突变是否会产生冲突就是其中一个非常重要的命题，因此可以使用当今相对比较成熟的机器学习模型根据现有样本对基因突变是否产生冲突进行预测，从而支持基因生物学研究的进一步发展。本次实验就是使用Kaggle网站提供的数据集通过使用SVM分类算法完成这样的任务。
# 二、数据集说明
本次实验中使用到的数据集来源于在数据科学业内享有盛名的数据科学竞赛平台Kaggle中的一个数据集Genetic Variant Classifications，该数据集由Kevin Arvai和Carlos Borroto在两年前上传。它是一个公共资源，包含有关人类遗传变异的解释。这些变异由临床实验室（通常是手动的）进行分类的，像良性、不确定意义、可能致病性和致病性等。当临床医生或研究人员试图解释变异是否对给定患者的疾病产生影响时，不同的人对于的变异冲突可能就会引起混淆。如果变异出现可能良性或良性、VUS、可能致病或致病就认为是冲突的，并且已经在CLASS列中标出。

该数据集共包含65189行，46列变量信息，以下表格对不同字段进行了解释。

|属性|解释（谷歌翻译）|
|:---|:---|
CHROM|	变异体所在的染色体
POS|	变异体在染色体上的位置
REF|	参考等位基因
ALT|	交替等位基因
AF_ESP	|来自 GO-ESP 的等位基因频率
AF_EXAC|	来自 ExAC 的等位基因频率
AF_TGP|	来自 1000 个基因组计划的等位基因频率
CLNDISDB|	疾病数据库名称和标识符的标记值对，例如 OMIM:NNNNNN
CLNDISDBINCL|	来自 GO-ESP 的等位基因频率
CLNDN|	ClinVar| 对于 CLNDISDB 中疾病标识符指定的概念的首选疾病名称
CLNDNINCL|	对于包含的变体：ClinVar 的首选疾病名称，用于 CLNDISDB 中疾病标识符指定的概念
CLNHGVS|	顶级（主要程序集、alt 或补丁）HGVS 表达式。
CLNSIGINCL|	包含该变体的单倍型或基因型的临床意义。报告为成对的 VariationID：临床意义。
CLNVC|	变体类型
CLNVI	|变异的临床来源报告为数据库和变异标识符的标签值对
MC|	序列本体 ID|molecular_consequence 形式的分子结果的逗号分隔列表
ORIGIN|	等位基因起源。可以添加以下一个或多个值： 0 - 未知；1 - 种系; 2 - 躯体; 4 - 继承的；8 - 父系；16 - 产妇；32 - 从头; 64 - 双亲；
SSR|	变体可疑原因代码。可以添加以下一个或多个值：0 - 未指定，1 - Paralog，2 - byEST，4 - oldAlign，8 - Para_EST，16 - 1kg_failed，1024 - 其
CLASS|	目标类的二进制表示。0 代表无冲突提交，1 代表冲突提交。
Allele|	用于计算结果的变异等位基因
Consequence	|结果类型：https://useast.ensembl.org/info/genome/variation/prediction/predicted_data.html#consequences
IMPACT|	后果类型的影响修饰符
SYMBOL|	基因名称
Feature_type|	特征类型。目前是 Transcript、RegulatoryFeature、MotifFeature 之一。
Feature|	特征的集成稳定 ID
BIOTYPE|	转录本或调控特征的生物型
EXON|	外显子数（总数中）
INTRON	|内含子数（总数中）
cDNA_position	|cDNA序列中碱基对的相对位置
CDS_position|	碱基对在编码序列中的相对位置
Protein_position|	氨基酸在蛋白质中的相对位置
Amino_acids	|仅当变异影响蛋白质编码序列时才给出
Codons	|带有大写变体碱基的替代密码子
DISTANCE	|从变体到转录本的最短距离
STRAND	|定义为 +（正向）或 -（反向）。
BAM_EDIT|	指示使用 BAM 文件编辑成功或失败
SIFT	|SIFT 预测和/或分数，两者都作为预测（分数）给出
PolyPhen|	PolyPhen 预测和/或分数
MOTIF_NAME|	在该位置对齐的转录因子结合谱的来源和标识符
MOTIF_POS|	对齐的TFBP中变化的相对位置
HIGH_INF_POS	|一个标志，表明该变体是否落在转录因子结合谱 (TFBP) 的高信息位置
MOTIF_SCORE_CHANGE|	TFBP 参考序列和变异序列的基序得分差异
LoFtool	|损失函数变体的函数损失容忍度得分：https://github.com/konradjk/loftee
CADD_PHRED	|Phred 标度的 CADD 分数
CADD_RAW|	变体有害性评分：http://cadd.gs.washington.edu/
BLOSUM62	|见：http://rosalind.info/glossary/blosum62/
# 三、实验目的
使用Python数据分析的相关技术和机器学习的相关算法，针对Kaggle中的一个数据集Genetic Variant Classifications，从不同角度分析遗传变异的各种特征以及对是否冲突的影响，并且根据这些特征主要使用各种算法进行建模，选取最佳模型进行评估，从而对划分出的测试集中的遗传变异是否会产生冲突进行预测。
# 四、实验环境
操作系统：Windows 10

编译器：Jupyter Notebook 6.3.0

Python：3.8.8

CPU：intell 11st i7

显卡：NVIDIA GeForce RTX 3070

其他模块：sklearn、scipy、numpy、missingno、random、warnings、operator、pandas、seaborn、scikitplot等
