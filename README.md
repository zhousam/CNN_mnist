# CNN_mnist
结合吴恩达卷积教学视频，进行基于TF的CNN的建立，利用mnist数据集进行验证
use Cnn to train mnist and test

# Learning From -- 1 code: https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/401_CNN.py
# Learning From -- 2 mnist Message: https://blog.csdn.net/zchang81/article/details/61918577
# Learning From -- 3 CNN Base Knowledge Video: http://mooc.study.163.com/course/2001281004?tid=2001392030#/info
# """
# Attention -> This code is only for python 3+.
# """


使用说明：
 模型的设计位于“CNN_model.py”中
 使用方法1：
   运行 “cnn_test_InOneFile.py”，即可得到训练中验证分类精度，以及测试集的分类度。
   
 使用方法2：
   A. 运行 “CNN_train.py”，可得到训练中验证集的分类度，并且保存训练的CNN模型，保存在“global_variable.py”里面的“save_path”目录下；
   B. 运行 “CNN_Test_FromModel.py”，可得到测试集的分类度。
