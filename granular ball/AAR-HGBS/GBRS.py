import copy
import csv
import time
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
warnings.filterwarnings("ignore")

#已知三条边，三角形面积对比
#余弦确定，角度确定样本点的相对方向与范围


start_time=time.time()
class GranularBall:
	def __init__(self, data, attribute):
		self.data = data[:, :]#全部数据
		self.attribute = attribute#当前的属性列
		self.data_no_label = data[:, attribute]#当前属性列数据
		self.num, self.dim = self.data_no_label.shape#当前属性列数据的行数与列数
		#print('num',self.num)
		self.center = self.data_no_label.mean(0)#获取当前属性列的中心
		self.label, self.purity, self.r = self.__get_label_and_purity_and_r()#获取标签，纯度和半径

	def __get_label_and_purity_and_r(self):#获取球的标签，纯度和半径
		count = Counter(self.data[:, -2])#获取标签类数量
		label = max(count, key=count.get)#将数量最多的标签类作为球的标签，并获取其数量
		purity = count[label] / self.num #获取该球的纯度
		distances = np.linalg.norm(self.data_no_label - self.center, axis=1)#计算每个点距离中心的欧式距离
		r = np.max(distances)# 半径取最大值
		return label, purity, r

	def split_2balls(self):
		labs = set(self.data[:, -2].tolist())#获取标签类的种类
		# print("labs",labs)
		i1 = 9999
		ti1 = -1
		ti0 = -1
		Balls = []# 用来存储分类的两个粒球
		i0 = 9999
		dol = np.sum(self.data_no_label, axis=1) #对每一行求和
		#print('dol',dol)
		if len(labs) > 1:#如果颗粒球里的标签类种类大于1，对颗粒球进行分裂
			for i in range(len(self.data)):#对每个样本点进行遍历，最后结果是两个不同标签类中与原点距离最近的两个点为新的球心
				if self.data[i, -2] == 1 and dol[i] < i1: #如果样本点的标签为 1，并且求和数小于il
					i1 = dol[i] #当前距离赋值给il
					ti1 = i #样本编号赋给ti1
				elif self.data[i, -2] !=1 and dol[i] < i0: #如果样本点的标签不为 1，并且求和数小于il)
					i0 = dol[i] #当前样本点距离赋值给i0
					ti0 = i #样本编号赋给ti0
			ini = self.data_no_label[[ti0, ti1], :]#取出两个新聚类中心的属性数据
			clu = KMeans(n_clusters=2,init=ini).fit(self.data_no_label)  #以当前属性列，构造两个聚类，聚类中心为新确定的两个样本点
			label_cluster = clu.labels_ #获取聚类后每个样本点的标签
			#print('label',label_cluster)
			if len(set(label_cluster)) > 1:#如果分裂成功，存在两个不同标签
				ball1 = GranularBall(self.data[label_cluster == 0, :], self.attribute)#对标签类为0的数据提取构造粒球
				ball2 = GranularBall(self.data[label_cluster == 1, :], self.attribute)#对标签类为1的数据提取构造粒球
				Balls.append(ball1)#添加球1
				Balls.append(ball2)#添加球2
			else:
				Balls.append(self)
		else:

			Balls.append(self)
		return Balls

def funtion(ball_list,minsam):#2个参数:颗粒球，最小样本量(这里设置为1)
	Ball_list = ball_list
	Ball_list = sorted(Ball_list, key=lambda x: -x.r, reverse=True)#根据半径r降序
	ballsNum = len(Ball_list)#颗粒球的数量
	j = 0
	ball = []
	while True:
		if len(ball) == 0:
			ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num])#添加第一个球的球心，半径，标签，样本数
			j += 1 #j+1
		else:
			flag = False
			for index, values in enumerate(ball):
				if values[2] != Ball_list[j].label and (
						np.sum((values[0] - Ball_list[j].center) ** 2) ** 0.5) < (
						values[1] + Ball_list[j].r) and Ball_list[j].r > 0 and Ball_list[j].num >= minsam / 2 and \
						values[3] >= minsam / 2:#如果球的标签不同,并且两个点的球心距离小于两个球的半径和(两个球有交叉)，并且,两个颗粒球中样本点数量都大于最小样本点
					balls = Ball_list[j].split_2balls()#满是上面条件时，对第j个颗粒球进行分裂
					if len(balls) > 1:#如果颗粒球分裂成功
						Ball_list[j] = balls[0]#分裂的第一个颗粒球作为原来第j个颗粒球
						Ball_list.append(balls[1])#分裂的第二个颗粒球添加到颗粒球中
						ballsNum += 1#颗粒球数量加1
					else:
						Ball_list[j] = balls[0]#分裂没成功，还是原来颗粒球

			if flag == False:
				# print(8)
				ball.append([Ball_list[j].center, Ball_list[j].r, Ball_list[j].label, Ball_list[j].num])#将第j个颗粒球添加到ball列表中
				j += 1#j+1
		if j >= ballsNum:#如果j大于颗粒球的数量，结束循环
			break
	return Ball_list

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
		if Ball_num1 == Ball_num2:  # 知道分裂前后颗粒球数量没有变化
			break
	return Ball_list

class GBList:
	def __init__(self, data=None, attribu=[]):
		self.data = data[:, :]#全部数据
		self.attribu = attribu#当前选择的属性列
		self.granular_balls = [GranularBall(self.data, self.attribu)]  # 将数据与属性列导入，获得颗粒球半径，标签，纯度等信息

	def init_granular_balls(self, purity=0.996, min_sample=1):
		ll = len(self.granular_balls)#粒球的数量
		i = 0
		while True:
			if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:#如果选择的颗粒球纯度小于阈值，并且数量大于最小样本
				split_balls = self.granular_balls[i].split_2balls()#对颗粒球进行分裂，返回的是颗粒球的数量
				if len(split_balls) > 1:#如果颗粒球成功分裂
					self.granular_balls[i] = split_balls[0]#用分裂的颗粒球中第一个替代原来第i个颗粒球
					self.granular_balls.append(split_balls[1])#将分裂的颗粒球中第二个添加到颗粒球
					ll += 1#颗粒球的数量加1
				else:
					i += 1#颗粒球加1，判断下一个颗粒球是否需要分裂
			else:
				i += 1#颗粒球加1，判断下一个颗粒球是否需要分裂
			if i >= ll:
				break
		ball_lists=self.granular_balls
		Bal_List=overlap_resolve(ball_lists, self.data, self.attribu,min_sample)#对颗粒球进行去重，4个参数:颗粒球，数据，选择的属性，最小样本量(这里设置为1)
		self.granular_balls=Bal_List#分裂后的颗粒球列表
		self.get_data()#获得所有数据
		self.data = self.get_data()

	def get_data_size(self):
		return list(map(lambda x: len(x.data), self.granular_balls))

	def get_purity(self):
		return list(map(lambda x: x.purity, self.granular_balls))

	def get_center(self):
		"""
		:return: the center of each ball.
		"""
		return np.array(list(map(lambda x: x.center, self.granular_balls)))

	def get_r(self):
		"""
		:return: return radius r
		"""
		return np.array(list(map(lambda x: x.r, self.granular_balls)))

	def get_data(self):#获取数据,来自 GBlist 中所有现有粒球的数据。
		"""
		:return: Data from all existing granular balls in the GBlist.
		"""
		list_data = [ball.data for ball in self.granular_balls]#列表中是ball的data属性，也就是全部数据
		return np.vstack(list_data)

	def del_ball(self, purty=0., num_data=0):
		#delete ball
		T_ball = []
		for ball in self.granular_balls:
			if ball.purity >= purty and ball.num >= num_data:
				T_ball.append(ball)
		self.granular_balls = T_ball.copy()
		self.data = self.get_data()

	def R_get_center(self, i):
		#get ball's center
		attribu = self.attribu
		attribu.append(i)
		centers = []
		for ball in range(self.granular_balls):
			center = []
			data_no_label = ball.data[:, attribu]
			center = data_no_label.mean(0)
			centers.append(center)
		return centers

def attribute_reduce(data, pur=1, d2=2):
	bal_num = -9999
	attribu = []
	re_attribu = [i for i in range(len(data[0]) - 2)]#获取属性列
	while len(re_attribu):#遍历所有属性
		N_bal_num = -9999
		N_i = -1
		N_attribu = copy.deepcopy(attribu)#复制
		for i in re_attribu:#依此添加属性
			N_attribu = copy.deepcopy(attribu)#复制
			N_attribu.append(i)#添加此属性
			print("N_attribu",N_attribu)
			gb = GBList(data, N_attribu)#引入全部数据和当前的属性到GBList
			gb.init_granular_balls(purity=pur, min_sample=2*(len(data[0])-d2))  # 纯度设置为1，最小颗粒球数设置为(属性列数量-2)的两倍
			ball_list1 = gb.granular_balls #
			Pos_num = 0
			for ball in ball_list1:#对颗粒球列表遍历
				if ball.purity >= 1:#如果颗粒球的纯度大于等于1
					Pos_num += ball.num #代表是正域
			if Pos_num > N_bal_num:#
				N_bal_num = Pos_num#当前正域赋值给Pos_num
				N_i = i #属性列表
		if N_bal_num >= bal_num:#当前正域最大值
			bal_num = N_bal_num#将当前正域赋值给bal_num
			attribu.append(N_i)#添加此属性
			re_attribu.remove(N_i)#删除此属性
		else:
			return attribu
	return attribu
def mean_std(a):
    # calucate average and standard
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std


if __name__ == "__main__":
	datan = ["Glass"] #数据集
	file = '..\..\DataSet\wine1.csv'
	for name in datan:
		with open(r"Result" + name + "0624.csv", "w", newline='', encoding="utf-8") as jg:
			writ = csv.writer(jg) #建立写入数据对象
			df = pd.read_csv(file) #数据集地址
			# 替换无穷大值为NaN
			df.replace([np.inf, -np.inf], np.nan, inplace=True)
			# 删除包含NaN值的行
			df.dropna(inplace=True)
			df['class'], _ = pd.factorize(df['class'])	#class列离散化
			#print(df['class'].unique())
			last_column = df['class'].copy() # 获取标签列
			first_column = df.iloc[:, 0].copy()  # 获取第一列
			# 将标签列和第一列互换位置
			df.iloc[:, 0] = last_column
			df['class'] = first_column
			data = df.values #数据集的数据，不带第一行标签行
			numberSample, numberAttribute = data.shape #获取数据的行和列
			minMax = MinMaxScaler()#特征数据缩放，归一化
			U = np.hstack((minMax.fit_transform(data[:, 1:]), data[:, 0].reshape(numberSample, 1)))#标签类转为二维数组，对非标签类进行归一化特征缩放,然后水平合并
			C = list(np.arange(0, numberAttribute - 1))#创建一个 从0开始，属性数量减1 结束的整数序列。并将序列转换成列表类型
			D = list(set(U[:, -1]))#set(U[:, -1],将标签类转化为集合，可以去除重复数据,然后再转化为列表类型
			sort_U = np.argsort(U[:, 0:-1], axis=0)#获得排序索引,每个标签类的数据的排序
			index = np.array(range(numberSample)).reshape(numberSample, 1)#生成二维数组,从0-样本数减1
			data_U = np.hstack((U, index))  # 在最后一列添加一个索引
			purty = 1 #设置纯度
			clf = KNeighborsClassifier(n_neighbors=1)#设置knn分类器
			orderAttributes = U[:, -1]#标签类
			mat_data = U[:, :-1]#属性类
			has_nan = np.isnan(mat_data).any()
			has_inf = np.isinf(mat_data).any()
			print(mat_data.shape)
			if has_nan or has_inf:
				print("数据集中存在NaN或无穷大值。")
			maxavg = -1
			maxStd = 0
			maxRow = []
			attribute_num = []
			print('data',data_U)
			for i in range((int)(numberAttribute)):
				nums = i
				Row = attribute_reduce(data_U, pur=purty, d2=nums)#获得正域最大的属性列表，区别是最小样本量不一样大
				attribute_num.append(Row)
				writ.writerow(["FGBNRS", Row])
				print("Row:", Row)
				mat_data = U[:, Row]
			# 	print('order',orderAttributes)
				scores = cross_val_score(clf, mat_data, orderAttributes, cv=10)
				avg, std = mean_std(scores) #返回平均值与方差
				writ.writerow(["min_sam",  Row, scores, avg])
				if maxavg < avg:
					maxavg = avg
					maxStd = std
					maxRow = copy.deepcopy(Row)
			print(maxRow)
			#print(U[:, maxRow])
			print("pre", maxavg, std)
	with open('..\GBRS_attribute\GBRS.txt', 'a') as file1:  # 使用'w'模式会覆盖文件内容，如果需要追加则使用'a'
		file1.write(f"{file}:\n")
		file1.write(f"{attribute_num}\n")

end_time=time.time()
print("代码运行时间: ", end_time - start_time, "秒")

