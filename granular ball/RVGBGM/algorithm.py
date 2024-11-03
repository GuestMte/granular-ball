import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression # 导入逻辑回归类
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
#Knn原始分类器 94.38 随机数 42
#LR原始分类器 97.68 随机数 99
#LR属性约简分类 97.75 随机数 99 个数:6个 属性：['Flavanoids', 'Alcohol', 'wines', 'Color', 'Proline', 'Hue']
#KNN原始分类器 98.33 随机数 42 个数 7个 属性:['Flavanoids', 'wines', 'Color', 'Alcohol', 'Proline', 'Nonflavanoid', 'Hue']
#平均值顺序排序结果：'Flavanoids','wines','Color','Alcohol','Total','Proline','Proanthocyanins','Nonflavanoid','Alcalinity','Magnesium','Malic','Ash','Hue',
data = pd.read_csv(r'../data/wine.csv')
data['class'], _ = pd.factorize(data['class'])
# 确定聚类簇数，即类别的种类数
num_clusters = len(data['class'].unique())
data_scaled = preprocessing.minmax_scale(data)
# 用标准化后的数据替换原始数据
data = pd.DataFrame(data_scaled, columns=data.columns)
data['class'], _ = pd.factorize(data['class'])
#print(data)
#确定初始聚类中心为属性值均值
X = data.drop('class', axis=1)
y = data['class']
m=X.groupby(y).mean()


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
        radius = np.mean(distances)
        # 取最大的距离作为该聚类的半径
        #radius = np.max(distances)
        # 将半径添加到列表中
        radii.append(radius)
    # 返回半径列表
    return radii

def cluster_er(X, labels, centers):
    ers = []
    eri=[]
    # 调用cluster_radius函数，获取每个聚类的半径列表
    radii = cluster_radius(X, labels, centers)
    # 遍历每个聚类中心
    for i in range(len(centers)):
        eri = []
    # 初始化该聚类的ER值为0
        er = 0 # 遍历其他聚类中心
        for j in range(len(centers)):
            if i != j: # 排除自身聚类 # 计算该聚类中心到其他聚类中心的欧氏距离
                distance = np.linalg.norm(centers[i] - centers[j]) # 累加该聚类半径与其他聚类半径之比除以距离之和
                print('er值:',(radii[i] + radii[j]) / distance)
                er_test= (radii[i] + radii[j]) / distance # 将ER值添加到列表中
                if er_test<=1:
                    er_append=0
                    print('erappend=',er_append)
                else:
                    er_append=er_test
                er += er_append
        ers.append(er)
    print(ers)
    er_sum = sum(ers)
    print('ers值', er_sum,end='')
    return er_sum


test_add=[]
for z in np.arange(0.8, 1.01, 0.01):
    accuracies = []
    feature_list = ['Flavanoids', 'wines', 'Color', 'Alcohol', 'Total', 'Proline', 'Proanthocyanins', 'Nonflavanoid',
                    'Alcalinity', 'Magnesium', 'Malic', 'Ash', 'Hue']
    selected_features = feature_list[0]
    m_selected = m[[selected_features]]
    print(m_selected)
    initial_centers = np.array(list(m_selected.values))
    labels = data['class'].values
    # print("类别标签")
    # print(labels)
    best_rail = cluster_er(data[[selected_features]].values, labels, initial_centers)
    print('属性', selected_features)
    selected_features = [selected_features]
    unselected_features = feature_list[1:]
    added_new_feature = True
    while added_new_feature:
        for feature in unselected_features:
            print(f'Average accuracy with {len(selected_features)} features: {best_rail}')
            print(selected_features)
            # 尝试添加这个特征到已经选择的特征列表中
            new_feature = selected_features +[feature]
            #print(new_feature)
            m_selected = m[new_feature]
            #print(m_selected)
            initial_centers = np.array(list(m_selected.values))
            labels = data['class'].values
            print("类别标签")
            print(labels)
            new_radii=cluster_er(data[new_feature],labels, initial_centers)
            if best_rail == 0:
                selected_features.append(feature)
                best_rail = new_radii
                unselected_features.remove(feature)
                added_new_feature = True
                print('unselect', unselected_features)
                break
            elif new_radii*z <= best_rail:
                # 保留这个特征，并更新最佳准确率
                selected_features.append(feature)
                best_rail = new_radii
                unselected_features.remove(feature)
                added_new_feature = True
                print('unselect',unselected_features)
                break
            else:
                added_new_feature=False
                continue
        if not unselected_features:  # 检查unselected_features是否为空
            break
    print(f'Final accuracy with {len(selected_features)} features: {best_rail:.4f}')
    print(f'Selected features: {selected_features}')
    test_add.append(selected_features)
highest_accuracy = 0
best_feature = None
accuracy_list_mean_LR=[]
accuracy_list_mean_knn=[]
for i in test_add:
    #print(i)
    # 读取数据
    data = pd.read_csv(r'../data/wine.csv')
    X = data[i]
    print(X)
    y = data.iloc[:, 0] # 标签数据
    #print(y)
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 10 折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []

    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # #knn分类器
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # 计算准确率、精确率和召回率
        accuracy = accuracy_score(y_test, y_pred)
        # 将结果存入列表
        accuracies.append(accuracy)
    average_accuracy = np.mean(accuracies)
    accuracy_list_mean_knn.append(average_accuracy)

    kf = KFold(n_splits=10, shuffle=True, random_state=99)
    accuracies = []
    precisions = []
    recalls = []

    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练逻辑回归分类器，并设置solver='liblinear'
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        # 计算准确率、精确率和召回率
        accuracy = accuracy_score(y_test, y_pred)
        # 将结果存入列表
        accuracies.append(accuracy)
    average_accuracy = np.mean(accuracies)
    accuracy_list_mean_LR.append(average_accuracy)




print('KNN长度:',len(accuracy_list_mean_knn))
print(accuracy_list_mean_knn)
plt.plot(np.arange(0.8, 1.01, 0.01), accuracy_list_mean_knn,label='RVGBGM-A-KNN',color='blue') # 画出z值和准确率的关系
print('LR长度:',len(accuracy_list_mean_LR))
print(accuracy_list_mean_LR)
plt.plot(np.arange(0.8, 1.01, 0.01), accuracy_list_mean_LR,label='RVGBGM-A-LR',color='green') # 画出z值和准确率的关系

# 读取数据集
data = pd.read_csv(r'../data/wine.csv')
data['class'], _ = pd.factorize(data['class'])
# 确定聚类簇数，即类别的种类数
num_clusters = len(data['class'].unique())
data_scaled = preprocessing.minmax_scale(data)
# 用标准化后的数据替换原始数据
data = pd.DataFrame(data_scaled, columns=data.columns)
data['class'], _ = pd.factorize(data['class'])
#print(data)
#确定初始聚类中心为属性值均值
X = data.drop('class', axis=1)
y = data['class']
m=X.groupby(y).mean()


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
        distances.sort()
        # 取距离中心最近的前90%的数据点之间的最大距离作为该聚类的半径
        radius = distances[int(len(distances)*0.99)]
        # 取最大的距离作为该聚类的半径
        #radius = np.max(distances)
        # 将半径添加到列表中
        radii.append(radius)
    # 返回半径列表
    return radii

def cluster_er(X, labels, centers):
    ers = []
    eri=[]
    # 调用cluster_radius函数，获取每个聚类的半径列表
    radii = cluster_radius(X, labels, centers)
    # 遍历每个聚类中心
    for i in range(len(centers)):
        eri = []
    # 初始化该聚类的ER值为0
        er = 0 # 遍历其他聚类中心
        for j in range(len(centers)):
            if i != j: # 排除自身聚类 # 计算该聚类中心到其他聚类中心的欧氏距离
                distance = np.linalg.norm(centers[i] - centers[j]) # 累加该聚类半径与其他聚类半径之比除以距离之和
                print('er值:',(radii[i] + radii[j]) / distance)
                er_test= (radii[i] + radii[j]) / distance # 将ER值添加到列表中
                if er_test<=1:
                    er_append=0
                    print('erappend=',er_append)
                else:
                    er_append=er_test
                er += er_test
        ers.append(er)
    print(ers)
    er_sum = sum(ers)
    print('ers值', er_sum,end='')
    return er_sum


test_add=[]
for z in np.arange(0.8, 1.01, 0.01):
    accuracies = []
    feature_list = ['Flavanoids', 'wines', 'Color', 'Alcohol', 'Total', 'Proline', 'Proanthocyanins', 'Nonflavanoid',
                    'Alcalinity', 'Magnesium', 'Malic', 'Ash', 'Hue']
    selected_features = feature_list[0]
    m_selected = m[[selected_features]]
    print(m_selected)
    initial_centers = np.array(list(m_selected.values))
    labels = data['class'].values
    # print("类别标签")
    # print(labels)
    best_rail = cluster_er(data[[selected_features]].values, labels, initial_centers)
    print('属性', selected_features)
    selected_features = [selected_features]
    unselected_features = feature_list[1:]
    added_new_feature = True
    while added_new_feature:
        for feature in unselected_features:
            print(f'Average accuracy with {len(selected_features)} features: {best_rail}')
            print(selected_features)
            # 尝试添加这个特征到已经选择的特征列表中
            new_feature = selected_features +[feature]
            #print(new_feature)
            m_selected = m[new_feature]
            #print(m_selected)
            initial_centers = np.array(list(m_selected.values))
            labels = data['class'].values
            print("类别标签")
            print(labels)
            new_radii=cluster_er(data[new_feature],labels, initial_centers)
            if new_radii*z <= best_rail:
                # 保留这个特征，并更新最佳准确率
                selected_features.append(feature)
                best_rail = new_radii
                unselected_features.remove(feature)
                added_new_feature = True
                print('unselect',unselected_features)
                break
            else:
                added_new_feature=False
                continue
        if not unselected_features:  # 检查unselected_features是否为空
            break
    print(f'Final accuracy with {len(selected_features)} features: {best_rail:.4f}')
    print(f'Selected features: {selected_features}')
    test_add.append(selected_features)
highest_accuracy = 0
best_feature = None
accuracy_list_max_knn=[]
accuracy_list_max_LR=[]
for i in test_add:
    #print(i)
    # 读取数据
    data = pd.read_csv(r'../data/wine.csv')
    X = data[i]
    print(X)
    y = data.iloc[:, 0] # 标签数据
    #print(y)
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 10 折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []

    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # 计算准确率、精确率和召回率
        accuracy = accuracy_score(y_test, y_pred)
        # 将结果存入列表
        accuracies.append(accuracy)
    average_accuracy = np.mean(accuracies)
    accuracy_list_max_knn.append(average_accuracy)

    kf = KFold(n_splits=10, shuffle=True, random_state=99)
    accuracies = []
    precisions = []
    recalls = []
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 训练逻辑回归分类器，并设置solver='liblinear'
        lr = LogisticRegression(solver='liblinear')
        lr.fit(X_train, y_train)
        #knn分类器

        # 预测测试集标签
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 将结果存入列表
        accuracies.append(accuracy)
    average_accuracy = np.mean(accuracies)
    accuracy_list_max_LR.append(average_accuracy)


print('LR长度:', len(accuracy_list_max_LR))
print(accuracy_list_max_LR)
plt.plot(np.arange(0.8, 1.01, 0.01), accuracy_list_max_LR,label='RVGBGM-B-LR',color='yellow')  # 画出z值和准确率的关系
print('knn长度:', len(accuracy_list_max_knn))
print(accuracy_list_max_knn)
plt.plot(np.arange(0.8, 1.01, 0.01), accuracy_list_max_knn,label='RVGBGM-B-KNN',color='red')


plt.legend()
plt.xlabel('λ')  # 设置x轴的标签
plt.ylabel('accuracy')  # 设置y轴的标签
#plt.title('Accuracy vs z value')  # 设置标题
plt.show()  # 显示图像
# # 输出平均准确率
# print(f'Average accuracy: {np.mean(accuracies):.4f}')