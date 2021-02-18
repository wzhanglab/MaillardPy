import numpy as np
from scipy.misc import derivative
import matplotlib.pylab as plt
import pandas as pd
import math
#from itertools import combinations  
from sklearn.model_selection import KFold
#from sklearn.preprocessing import MinMaxScaler

#
def identy(x):
    return x

def forward(xin,connt,actfc):
    
    x_d1,x_d2 = xin. shape
    c_d1,c_d2 = connt.shape
    assert x_d1+1==c_d1

    zin = np.vstack((xin,np.ones((1,x_d2))))
    a = np.array(np.mat(connt).T * np.mat(zin))
    z = actfc(a)
 
    return a,z
 
def backward(dtin,ain,zin,connt,actfc):
    d_d1,d_d2 = dtin.shape
    a_d1,a_d2 = ain.shape
    z_d1,z_d2 = zin.shape
    c_d1,c_d2 = connt.shape
    assert (d_d1==c_d2) & (a_d1==z_d1==c_d1-1)
    
    dt = np.array(np.mat(connt[0:c_d1-1,:]) * np.mat(dtin))
    dt = derivative(actfc,ain,order=5)*dt
 
    z = np.vstack((zin,np.ones((1,z_d2))))
 
    e = np.zeros((c_d1,c_d2))
    for i in range(0,c_d1):
        e[i] = np.sum(z[i]*dtin,axis=1)/d_d2
    ##print(e.shape)
    return e,dt
def calc_corr(a,b):
    E_a = np.mean(a)
    E_b = np.mean(b)
    E_ab=np.mean(list(map(lambda x:x[0]*x[1],zip(a,b))))

    # 计算分子，协方差—cov(a,b)=E(ab)-E(a)*E(b)
    cov_ab = E_ab - E_a * E_b

    def square(lst):
        res=list(map(lambda x:x**2,lst))
        return res

    # 计算分母，D(X)=E(X²)-E²(X)
    D_a=np.mean(square(a))-E_a**2
    D_b=np.mean(square(b))-E_b**2

    σ_a=np.sqrt(D_a)
    σ_b=np.sqrt(D_b)

    corr_factor = cov_ab / (σ_a*σ_b)
    return corr_factor

#构建网络
    #输入数据集
file_1=pd.read_csv('Bcolumn_1.csv')
file_tmp = file_1.values.tolist()
file_tmp= np.array(file_tmp)
file_tmp=file_tmp[:,1:]
temp_con=file_tmp[:,:63]
temp_label=file_tmp[:,63]
con_m = np.mean(temp_con)
con_mx=np.max(temp_con)
con_mn = np.min(temp_con)
label_m = np.mean(temp_label)
label_mx = np.max(temp_label)
label_mn = np.min(temp_label)
def normalize(file_tmp,mx,mn,m):
    return np.array([(i - m) / (mx - mn) for i in file_tmp])

def traverse_normalize(file_tmp,mx,mn,m):
    return np.array([i*(mx-mn)+m for i in file_tmp])

nor_con=normalize(temp_con,con_mx,con_mn,con_m)
nor_label=normalize(temp_label,label_mx,label_mn,label_m).reshape(68,1)
#tranor_con=traverse_normalize(nor_con,con_mx,con_mn,con_m)
#tranor_label=traverse_normalize(nor_label,label_mx,label_mn,label_m)
Sam=np.hstack((nor_con,nor_label))


#K折交叉验证
count=0
num=1
New_sam=KFold(n_splits=5)
for train_index,test_index in New_sam.split(Sam):  # 对Sam数据建立5折交叉验证的划分
    Sam_train,Sam_test=Sam[train_index],Sam[test_index]
    print('训练集数量:',Sam_train.shape,'测试集数量:',Sam_test.shape) 
    
#p=list(range(6,12))#往后取一个数，行数
#count=0
#x=file_tmp[1:6,:63].astype(np.float64)
#for i in combinations(p, 3):
 #   for j in i:
  #      x_tmp=file_tmp[j,:63].reshape(1,63).astype(np.float64)
   #     x= np.vstack((x,x_tmp))
        

#print(count)

    t=Sam_train[:,63].reshape(1,-1).astype(np.float64)
    x=(Sam_train[:,:63].astype(np.float64)).T



    units = np.array([63,16,9,6,1])
    units_bias = units+1
##print(units_bias)
 #权重初始化
    connt = {}
    for i in range(1,len(units_bias)):
        connt[i] = np.random.uniform(-1,1,size=(units_bias[i-1],units[i]))
##print(connt)
 
    actfc = {0:identy}
    for i in range(1,len(units_bias)-1):
        actfc[i] = np.tanh
    actfc[len(units)-1] = np.tanh
##print(actfc)
    
    for k  in range(0,5000):
        a = {0:x}
        z = {0:actfc[0](a[0])}
        for i in range(1,len(units)):
            a[i],z[i] = forward(z[i-1],connt[i],actfc[i])
 
        dt = {len(units)-1:z[i]-t}
        e = {}
        for i in range(len(units)-1,0,-1):
            e[i],dt[i-1] = backward(dt[i],a[i-1],z[i-1],connt[i],np.tanh)
 
        pp = 0.05
        for i in range(1,len(units_bias)):
            connt[i] = connt[i]-pp*e[i]
    ##print(connt)
 #外部验证
    y=(Sam_test[:,:63].astype(np.float64)).T
    label=Sam_test[:,63].reshape(1,-1).astype(np.float64)
    a = {0:y}
    z = {0:actfc[0](a[0])}
    for i in range(1,len(units)):
        a[i],z[i] = forward(z[i-1],connt[i],actfc[i])
    
    err=z[i]-label
    sum=0
#print(len(err))
    for j in range(Sam_test.shape[0]):
        temp=err[0][j]*err[0][j]
        sum+=temp

    RMSE=math.sqrt( sum/Sam_test.shape[0] )
    print(RMSE)
    count+=RMSE
  # temp_connt=connt
   # for w in range(1,5):
    #    temp_connt[w]=list(temp_connt[w])
    #df=pd.DataFrame(temp_connt)
    #df.to_csv('Dataframe_{}.csv'.format(num))
    num+=1
    pre = traverse_normalize(z[4],label_mx,label_mn,label_m)
    label_1=traverse_normalize(label,label_mx,label_mn,label_m)
    corr=calc_corr(pre,label)
    print(corr)
    [_,len_label]=label_1.shape
    x_map=np.linspace(0,len_label,len_label)
    plt.scatter(x_map,label_1.reshape(len_label),color="red",label='label',linewidth=2)
    plt.plot(x_map,pre.reshape(len_label),color='green',label='predict',linewidth=2)
    plt.legend(loc='lower right')
    plt.show()
ACC=count/5
print(ACC)
