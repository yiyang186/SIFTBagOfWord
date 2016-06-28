#计算机视觉大作业

**实现下列系统之一，并文档说明介绍：**
1. SIFT + Bag-of-word 的图像检索系统
可参考这个PPT[Bag of visual words model: recognizing object categories](http://www.robots.ox.ac.uk/~az/icvss08_az_bow.pdf "Bag of visual words model: recognizing object categories")
2. 实现论文：
 - Y.Boykov and M.Jolly    Interactive Graph Cuts for Optimal Boundary and Region 
 Segmentation of Objects in N-D Images. ICCV 2001.(Image Segmentation)
 - Philip H. S. Torr, Andrew Zisserman:MLESAC: A New Robust Estimator with 
 Application to Estimating Image Geometry. Computer Vision and Image Understanding
  78(1): 138-156 (2000 ) (MLESAC + RANSAC)

##需要
python 2.7.11(Anaconda 2.5.0 (64-bit))
opencv 2.4.11
scikit-learn 0.17
numpy 1.10.4
scipy 0.17.0

##执行
```
python homework.py
```
baidu了一些图片作为trainset和test，准确率90%。

##感谢
重构并修改了[这份代码](https://github.com/bikz05/bag-of-words)，简化了部分功能，
并以descriptors的tf-idf作为训练特征重新构建分类器(原代码疏忽？。。。算了idf却只
用了tf。。。不知道他怎么想的，反正我看用tf-idf的效果比只用tf的好)。