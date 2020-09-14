import jieba    #第三方模块 jieba 用于进行中文分词
import numpy
import os
import psutil   #用于测试程序的运行时间、占用内存等性能
import sys
from sklearn.feature_extraction.text import CountVectorizer

#   Jaccard系数计算模块
#   输入两个string，输出一个int的系数
def Jaccard_Similarity(Str1, Str2):
    def Add_Space(Str):
        #   引入第三方模块jieba来进行中文分词
        Str_list = jieba.cut(Str, cut_all=False)
        #   把分词后的词用空格连接成string，以便转化成TF矩阵
        return ' '.join(list(Str_list))

    Str1, Str2 = Add_Space(Str1), Add_Space(Str2)
    #   记录词频，分词器为以' '为分隔符的lambda
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [Str1, Str2]
    #   转化为TF(Term Frequencies)矩阵
    vectors = cv.fit_transform(corpus).toarray()
    #   用第三方模块numpy求交并集
    numerator = numpy.sum(numpy.min(vectors,axis=0))    #   交集
    denominator = numpy.sum(numpy.max(vectors,axis=0))  #   并集
    #   返回Jaccard系数
    return 1.0 * numerator / denominator

#   文本输入函数，将文本从硬盘读入内存
def Readin(TextAddress):
    f = open(TextAddress,"r",encoding="UTF-8")
    Str = f.read()
    f.close()
    return Str

#   结果输出函数，将内存中的结果写入硬盘
def WriteOut(Jaccard, TextAddress):
    f = open(TextAddress,"w",encoding="UTF-8")
    f.write( str(Jaccard)[0:4] )
    f.close()

#   数据测试函数，测试10个orig文件，得出9个重复率
def Test():
    Str = Readin("d:/code/testdata/orig.txt")

    Str_add = Readin("d:/code/testdata/orig_0.8_add.txt")
    print("orig_0.8_add的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_add)))

    Str_del = Readin("d:/code/testdata/orig_0.8_del.txt")
    print("orig_0.8_del的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_del)))

    Str_dis1 = Readin("d:/code/testdata/orig_0.8_dis_1.txt")
    print("orig_0.8_dis1的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_dis1)))

    Str_dis3 = Readin("d:/code/testdata/orig_0.8_dis_3.txt")
    print("orig_0.8_dis3的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_dis3)))

    Str_dis7 = Readin("d:/code/testdata/orig_0.8_dis_7.txt")
    print("orig_0.8_dis7的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_dis7)))

    Str_dis10 = Readin("d:/code/testdata/orig_0.8_dis_10.txt")
    print("orig_0.8_dis10的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_dis10)))

    Str_dis15 = Readin("d:/code/testdata/orig_0.8_dis_15.txt")
    print("orig_0.8_dis15的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_dis15)))

    Str_mix = Readin("d:/code/testdata/orig_0.8_mix.txt")
    print("orig_0.8_mix的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_mix)))

    Str_rep = Readin("d:/code/testdata/orig_0.8_rep.txt")
    print("orig_0.8_rep的测试结果：{:f}".format(Jaccard_Similarity(Str, Str_rep)))

#   性能分析函数
def Analyze():
    print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) )
    print(u'当前进程的使用的CPU时间：%.4f s' % (psutil.Process(os.getpid()).cpu_times().user) )


Str_A = Readin(sys.argv[1])
Str_B = Readin(sys.argv[2])
num = Jaccard_Similarity(Str_A, Str_B)
WriteOut(num, sys.argv[3])

#   Test()
#   Analyze()
