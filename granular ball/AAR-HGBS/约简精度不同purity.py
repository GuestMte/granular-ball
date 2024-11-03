import copy
import csv
import time
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")



# 定义一个函数，用于计算每个聚类的半径
def cluster_radius(X, labels, centers):
    # 初始化一个空列表，用于存储每个聚类的半径
    radii = []
    # 遍历每个聚类中心
    for i in range(len(centers)):
        # 提取属于该聚类的数据点
        cluster_points = X[labels == i]
        # 计算该聚类中每个点到中心的欧氏距离
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
        # 取距离中心最近的前90%的数据点之间的最大距离作为该聚类的半径
        #radius = np.mean(distances)
        # 取最大的距离作为该聚类的半径
        radius = np.max(distances)
        # 将半径添加到列表中
        radii.append(radius)
    # 返回半径列表
    return radii

class GranularBall:
	def __init__(self, data, attribute):
		self.data = data[:, :]#全部数据
		self.attribute = attribute#当前的属性列
		self.data_no_label = data[:, attribute]#当前属性列数据
		self.num, self.dim = self.data_no_label.shape#当前属性列数据的行数与列数
		self.center = self.data_no_label.mean(0)#获取当前属性列的中心
		self.label, self.r,self.labelcount,self.purity= self.__get_label_and_purity_and_r()#获取标签，纯度和半径
		self.SD=1

	def __get_label_and_purity_and_r(self):#获取球的标签，纯度和半径
		count = Counter(self.data[:, -2])
		most_common_labels = count.most_common(1)
		label, labelcount = most_common_labels[0]
		#purity = count[label] / self.num
		purity= labelcount / self.num
		distances = np.linalg.norm(self.data_no_label - self.center, axis=1)#计算每个点距离中心的欧式距离
		r = np.max(distances)# 半径取最大值
		return label, r,labelcount,purity


	def split_2balls(self,number,er):
		labs = set(self.data[:, -2].tolist())#获取标签类的种类
		#print('lab',labs)
		# print("labs",labs)
		i1 =999999
		ti1 = -1
		ti0 = -1
		Balls = []# 用来存储分类的两个粒球
		i0 = 999999
		#dol = np.sum(self.data_no_label, axis=1) #对每一行求和
		dol = np.linalg.norm(self.data_no_label - self.center, axis=1)#不同样本点离之前球心最近距离为新的球心
		#print('dol',dol)
		if len(labs) > 1 :#如果颗粒球里的标签类种类大于1，对颗粒球进行分裂
			for i in range(len(self.data)):#对每个样本点进行遍历，最后结果是两个不同标签类中与原点距离最近的两个点为新的球心
				if self.data[i, -2] == 1 and dol[i] < i1: #如果样本点的标签为 1，并且求和数小于il
					i1 = dol[i] #当前距离赋值给il
					ti1 = i #样本编号赋给ti1
				elif self.data[i, -2] !=1 and dol[i] < i0: #如果样本点的标签不为 1，并且求和数小于il
					i0 = dol[i] #当前样本点距离赋值给i0
					ti0 = i #样本编号赋给ti0
			ini = self.data_no_label[[ti0, ti1], :]#取出两个新聚类中心的属性数据
			clu = KMeans(n_clusters=2,init=ini).fit(self.data_no_label)  #以当前属性列，构造两个聚类，聚类中心为新确定的两个样本点
			label_cluster = clu.labels_ #获取聚类后每个样本点的标签
			#print('label',label_cluster)
			ball_add=False
			if len(set(label_cluster)) > 1:#如果分裂成功，存在两个不同标签
				if len(self.data[label_cluster == 0, :])>number:
					#print('lennum111',len(self.data[label_cluster == 0, :]))
					ball1 = GranularBall(self.data[label_cluster == 0, :], self.attribute)#对标签类为0的数据提取构造粒球
					Balls.append(ball1)  # 添加球1
					ball_add =True
				if len(self.data[label_cluster == 1, :])>number:
					#print('lennum222', len(self.data[label_cluster == 1, :]))
					ball2 = GranularBall(self.data[label_cluster == 1, :], self.attribute)#对标签类为1的数据提取构造粒球
					Balls.append(ball2)  # 添加球2
					ball_add =True
				if ball_add==False:
					#print('True')
					Balls.append(self)
			else:
				#print('分裂不成功',666666)
				Balls.append(self)
		elif len(self.data)>1 and self.SD>1.4 :
			for i in range(len(self.data)):#对每个样本点进行遍历，最后结果是两个不同标签类中与原点距离最近的两个点为新的球心
				sorted_indices = np.argsort(dol)[::-1]  # 反转索引以获取降序排序
				top_two_indices = sorted_indices[:2]
				ti0=top_two_indices[0]
				ti1=top_two_indices[1]
			ini = self.data_no_label[[ti0, ti1], :]  # 取出两个新聚类中心的属性数据
			clu = KMeans(n_clusters=2, init=ini).fit(self.data_no_label)  # 以当前属性列，构造两个聚类，聚类中心为新确定的两个样本点
			label_cluster = clu.labels_
			ball_add = False
			if len(set(label_cluster)) > 1:  # 如果分裂成功，存在两个不同标签
				#print('SD分裂')
				if len(self.data[label_cluster == 0, :]) > number:
					ball1 = GranularBall(self.data[label_cluster == 0, :], self.attribute)  # 对标签类为0的数据提取构造粒球
					Balls.append(ball1)  # 添加球1
					ball_add = True
				if len(self.data[label_cluster == 1, :]) > number:
					ball2 = GranularBall(self.data[label_cluster == 1, :], self.attribute)  # 对标签类为1的数据提取构造粒球
					Balls.append(ball2)  # 添加球2
					ball_add = True
				if ball_add == False:
					print('True')
					Balls.append(self)
		else:
			Balls.append(self)
		return Balls

def funtion(ball_list,minsam):#2个参数:颗粒球，最小样本量(这里设置为1)
	Ball_list = ball_list
	Ball_list1=[]
	Ball_list = sorted(Ball_list, key=lambda x: -x.r, reverse=True)#根据半径r降序
	remaining_balls = Ball_list[:]
	# 使用嵌套的循环来比较每对球
	add_ball=True
	for i in range(len(Ball_list)):
		for j in range(i + 1, len(Ball_list)):
			ball1 = Ball_list[i]
			ball2 = Ball_list[j]
			# # 计算两个球心之间的距离
			distance = np.linalg.norm(ball1.center-ball2.center)
			# # 如果距离小于其中一个小球的半径，移除半径较小的球
			if distance < (ball2.r-ball1.r):
				add_ball=False
				remaining_balls.remove(Ball_list[i])
				break  # 无需继续检查与ball1的其它比较
		#这边要看一下
		# if add_ball==True:
		# 	Ball_list1.append(ball1)
			# 注意这里不需要break，因为j还会递增，并检查下一个球

	Ball_list1=sorted(remaining_balls, key=lambda x: -x.r, reverse=True)

	return Ball_list1

def overlap_resolve(ball_list, data, attributes_reduction,min_sam):#对颗粒球进行去重，4个参数:颗粒球，数据，选择的属性，最小样本量(这里设置为1)
	Ball_list = funtion(ball_list,min_sam)  #对不同标签重合颗粒球进行分裂
	while True:
		init_center = []
		Ball_num1 = len(Ball_list)  # 颗粒球的数量
		for i in range(len(Ball_list)):#遍历每个颗粒球
			# print('center',i)
			init_center.append(Ball_list[i].center)#存储每个球心
		ClusterLists = KMeans(init=np.array(init_center), n_clusters=len(Ball_list)).fit(data[:, attributes_reduction])#以每个球心为初始中心进行聚类
		data_label = ClusterLists.labels_#获取标签
		ball_list = []
		for i in set(data_label):#对每个聚类标签遍历
			ball_list.append(GranularBall(data[data_label == i, :], attributes_reduction))#为每个聚类构造颗粒球
		Ball_list = funtion(ball_list, min_sam)#对不同标签重合颗粒球进行分裂
		Ball_num2 = len(Ball_list)  # get ball numbers
		if Ball_num1 == Ball_num2:  # 直到分裂前后颗粒球数量没有变化
			break
	return Ball_list


class GBList:
	def __init__(self, data=None, attribu=[]):
		self.data = data[:, :]#全部数据
		self.attribu = attribu#当前选择的属性列
		self.data_no_label = data[:, attribu]
		self.num, self.dim = self.data_no_label.shape
		self.lebe=len(np.unique(self.data[:,-2]))
		# self.center = self.data_no_label.mean(0)
		self.granular_balls = [GranularBall(self.data, self.attribu)]  # 将数据与属性列导入，获得颗粒球半径，标签，纯度等信息


	def cluster_er(self):
		length=len(self.granular_balls)
		#print('类别',self.lebe)
		if length < self.lebe:
			for i in range(length):
				if len(set(self.granular_balls[i].data[:, -2].tolist())) > 1:
					#print('分裂的次数')
					self.granular_balls[i].SD=99999999
		# 遍历每个聚类中心
		else:
			for i in range(length):
				m=0
				sd = 0  # 遍历其他聚类中心
				for j in range(length):#球中多数类数量占球中所有样本点比重
					if i != j and (self.granular_balls[i].label != self.granular_balls[j].label or (len(set(self.granular_balls[i].data[:, -2].tolist()))>1 and (self.granular_balls[j].labelcount / self.granular_balls[j].num)<=0.8)):  # 排除自身聚类 # 计算该聚类中心到其他聚类中心的欧氏距离
						#print('m的值',m)
						distance = np.linalg.norm(self.granular_balls[i].center - self.granular_balls[j].center)  # 累加该聚类半径与其他聚类半径之比除以距离之和
						#print('sd值:', (self.granular_balls[i].r + self.granular_balls[j].r) / distance)
						sd_test = (self.granular_balls[i].r + self.granular_balls[j].r) / distance  # 将ER值添加到列表中
						if sd_test > 1:
							sd_true = sd_test
							m+=1
							#print('erappend=', sd_true)
							sd += sd_true
						# sd+=sd_test
				if m!=0:
					sd=sd/m
			# sd=sd/(length-1)
				self.granular_balls[i].SD=sd
		#print('er值', sd)
		return self.granular_balls


	def init_granular_balls(self,er):
		number=int(self.num/400)
		#print('er值',er)
		#number=1
		#print('number',number)
		ll = len(self.granular_balls)#粒球的数量
		self.granular_balls=self.cluster_er()
		i = 0
		while True:#先对原有颗粒球进行分裂，然后再对分裂后的颗粒球进行检查是否需要分裂，直到所有颗粒球都满足条件，即i(当前颗粒球的序号)>ll(当前颗粒球的数量)
			if self.granular_balls[i].SD > er :#如果选择的颗粒球纯度小于阈值，并且数量大于最小样本
				split_balls = self.granular_balls[i].split_2balls(number=number,er=er)#对颗粒球进行分裂，返回的是分裂后的颗粒球
				if len(split_balls) > 1:#如果颗粒球成功分裂
					#print('开始分裂')
					self.granular_balls[i] = split_balls[0]#用分裂的颗粒球中第一个替代原来第i个颗粒球
					self.granular_balls.append(split_balls[1])#将分裂的颗粒球中第二个添加到颗粒球
					self.granular_balls = self.cluster_er()
					ll += 1#颗粒球的数量加1
				else:
					#print('分裂没有成功')
					i += 1#如果颗粒球没有分裂成功，i+1
			else:
				i += 1#如果此颗粒球不需要分裂，则遍历下一个颗粒球
			if i >= ll:#如果i>ll,说明遍历完成了所有颗粒球
				break
		ball_lists=self.granular_balls #将当前所有颗粒球赋值给ball_list
		ball_lists = funtion(ball_lists, 1)
		self.granular_balls=ball_lists#分裂后的颗粒球列表
		self.granular_balls = self.cluster_er()
		self.get_data()#获得所有数据
		self.data = self.get_data()

	def get_data(self):  # 获取数据,来自 GBlist 中所有现有粒球的数据。
		"""
		:return: Data from all existing granular balls in the GBlist.
		"""
		list_data = [ball.data for ball in self.granular_balls]  # 列表中是ball的data属性，也就是全部数据
		return np.vstack(list_data)


def attribute_reduce(data,number,er,purity):
	print('总数量',number)

	er=er
	print('er1111',er)
	len_sum = 9999
	bal_num = -9999
	attribu = []
	print(data)
	re_attribu = [i for i in range(len(data[0]) - 2)]#获取属性列
	print(re_attribu)
	while len(re_attribu):#遍历所有属性
		N_bal_len = 9999
		N_bal_num = -9999
		N_i = -1
		for i in re_attribu:#依此添加属性
			N_attribu = copy.deepcopy(attribu)#复制
			N_attribu.append(i)#添加此属性
			print("N_attribu",N_attribu)
			gb = GBList(data, N_attribu)#引入全部数据和当前的属性到GBList
			gb.init_granular_balls(er=er)
			ball_list1 = gb.granular_balls
			Pos_num = 0
			for ball in ball_list1:#对颗粒球列表遍历
				if ball.purity >= purity:  # 如果颗粒球的纯度大于等于1
					Pos_num += ball.labelcount  # 代表是正域
			print('数量',Pos_num)
			if Pos_num > N_bal_num:  #
				N_bal_num = Pos_num  # 当前正域赋值给Pos_num
				N_i = i  # 属性列表
			if Pos_num==number:
				break
		if N_bal_num > bal_num:  # 当前正域最大值
			bal_num = N_bal_num  # 将当前正域赋值给bal_num
			print('目前数量',bal_num)
			print('覆盖率', bal_num/number)
			attribu.append(N_i)  # 添加此属性
			re_attribu.remove(N_i)  # 删除此属性
		else:
			return attribu
	return attribu

def mean_std(a):
    # calucate average and standard
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std


if __name__ == "__main__":
    datan = ["\wine1"]
    #desired_cols =['1','2','3','4','5','6','7','8','9']
    for name in datan:
        df = pd.read_csv(r"..\..\DataSet" + name + ".csv")
        number,col=df.shape
        #处理数据,包括归一化,去重,标签列移到最后一列，最后一列加索引等操作，最后得到一个data_U
        df.replace([np.inf, -np.inf], np.nan, inplace=True)# 替换无穷大值为NaN
        df.dropna(inplace=True)# 删除包含NaN值的行
        df['class'], _ = pd.factorize(df['class'])#标签类离散化
        cols = df.columns.tolist()
        cols.remove('class')
        cols.append('class')
        df = df[cols]
        print(df)
        data = df.values #
        numberSample, numberAttribute = data.shape#获取数据的行数与列数
        minMax = MinMaxScaler()#数据归一化处理
        U = np.hstack((minMax.fit_transform(data[:, :-1]), data[:, -1].reshape(numberSample, 1)))#归一化处理后堆叠在一起，次数标签类在最后面
        index = np.array(range(numberSample)).reshape(numberSample, 1)  # 生成二维数组索引,从0-样本数减1
        data_U = np.hstack((U, index))
        maxavg = -1
        maxStd = 0
        maxRow = []
        er1=0
        pur=[0.9,0.99]
        start_time=time.time()
        plt.axhline(y=94.9, color='r', linestyle='--', label='raw_data')
        for purity in pur:
            avg_list = []
            if purity==0.99:
                name='purity(HGBT)=0.99'
            else:
                name='purity(HGBT)=0.9'
            for er in np.arange(1,1.51, 0.05):
                Row = attribute_reduce(data_U,number,er,purity=purity)
                mat_data = U[:, Row]
				#mat_data = U[:,[0,1,2,3,4,5,6,7,8,9]]
                clf = KNeighborsClassifier(n_neighbors=1)  # 设置knn分类器
                orderAttributes = U[:, -1]  # 标签类
                scores = cross_val_score(clf, mat_data, orderAttributes, cv=10)
                avg, std = mean_std(scores)
                avg_list.append(avg*100)
            if purity==0.99:
                plt.plot(np.arange(1, 1.51, 0.05), avg_list, marker='*',label=name)
            else:
                plt.plot(np.arange(1, 1.51, 0.05), avg_list, marker='s', label=name)
            #plt.ylim(50, 100)
        plt.legend(loc='best')
        plt.xlabel('HSD(B)')  # 设置x轴的标签
        plt.ylabel('accuracy')  # 设置y轴的标签
        plt.show()
















