# %% 最小二乘法非线性函数拟合返回参数和拟合优度
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd

# 定义拟合函数
def func(p, x):
    a, b, c = p
    return a + b * x ** c #power 

# 定义误差函数
def error(p, x, y):
    return func(p, x) - y

# 生成数据
df = pd.read_excel('Interval and Z2.xlsx', sheet_name='6-2-yz-1', header=0, usecols='BL:BO', nrows=1001)
Iv = df.values[0:16,0]
z2 = df.values[0:16,2]
Iv_fit = np.linspace(0.05,10,200) 

# 初始参数
p0 = [0, 0, 0]

# 最小二乘法拟合
para, cov = leastsq(error, p0, args=(Iv, z2))

# 计算拟合优度值
ss_err = ((func(para,Iv)-z2.mean() )** 2).sum()
ss_tot = ((z2 - z2.mean()) ** 2).sum()
r_squared = ss_err / ss_tot

# 绘制图像
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig2 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,0.8)
ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,0.8,0.1))
ax.set_ylabel('Z2/mm')
ax.set_xlabel('Interval/mm')
# ax.set_title('Fruit supply by kind and color')
ax.grid(True,ls='--')
ax.scatter(Iv,z2,marker='o',color='red',s=50,label='test')
ax.plot(Iv_fit, para[0]+para[1]*Iv_fit**para[2], color='red', linestyle='solid', lw=2,label='fit')
ax.legend()

plt.show()
print(para)
print(f"拟合优度值: {r_squared:.6f}")
# %% 绘制轮廓线 **************************************************************
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_excel('sandstone_profile.xlsx', sheet_name='sin', header=0, usecols='A:J', nrows=1002)
x = df.values[0:936,4]
y = df.values[0:936,5]

sampling_interval = 2.5
interval_space = sampling_interval/0.05

x1 = x[0::int(interval_space)]
y1 = y[0::int(interval_space)]

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig1 = plt.figure(figsize=(24,2), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-5,55)
ax.set_ylim(-5,5)
ax.set_xticks(np.arange(0,55,5))
ax.set_yticks(np.arange(-5,5,2))
ax.set_ylabel('y/mm')
ax.set_xlabel('x/mm')
ax.grid(True,ls='--')
ax.scatter(x1,y1,marker='^',color='blue',s=10,label='profile')
ax.plot(x1,y1,color='blue',lw=2,label='profile')

plt.savefig('pf_6-2-yz.svg')
plt.show()
# %%  计算z2值与间隔的关系**************************************************************
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

def z2_cal(interval, x, y):


    space = interval/0.05
    k_arr=[]
    x1 = x[0::int(space)]
    y1 = y[0::int(space)]
    for i in range(len(x1)-1):

        k = ((y1[i+1]-y1[i])/(x1[i+1]-x1[i]))**2
        k_arr.append(k)
        i = i+1

    z_2 = np.average(k_arr)**0.5

    return z_2

def fit_z2_vs_interval(Iv,z2_list):

    def func(p, x):
        a, b = p
        return a + b * x #power 

    # 定义误差函数
    def error(p, x, y):
        return func(p, x) - y

    # 初始参数
    p0 = [0, 0]

    # 最小二乘法拟合
    para, cov = leastsq(error, p0, args=(Iv, z2_list))

    # 计算拟合优度值
    ss_err = ((func(para,Iv)-z2_list.mean() )** 2).sum()
    ss_tot = ((z2_list - z2_list.mean()) ** 2).sum()
    r_squared = ss_err / ss_tot

    return para, r_squared

df = pd.read_excel('sandstone_profile.xlsx', sheet_name='sin', header=0, usecols='A:J', nrows=1001)
x = np.array(df.values[0:1001,0])
y = np.array(df.values[0:1001,1])

Inter = np.linspace(0.05,10,200)
z2_list = []

for i in range(len(Inter)):

    z2 = z2_cal(Inter[i],x,y)
    z2_list.append(z2)
    i = i+1

z2_list = np.array(z2_list)

para, r_squared = fit_z2_vs_interval(Inter,z2_list)

print(para,f"拟合优度值: {r_squared:.6f}")

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,4)
ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,4,1))
ax.set_ylabel('Z2/mm')
ax.set_xlabel('Interval/mm')
ax.grid(True,ls='--')

ax.scatter(Inter,z2_list,marker='o',color='red',s=50,label='test')
ax.plot(Inter, para[0]+para[1]*Inter, color='red', linestyle='solid', lw=2,label='fit')
ax.legend()
plt.savefig('z2_sin.svg')
plt.show()
# %%  计算Grasselli值
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

def ag_cal(x_G, y_G, ag_space_G):

    ag_list = []

    for i in range(len(x_G)-1):

        ag = np.rad2deg(np.arctan((y_G[i+1]-y_G[i])/(x_G[i+1]-x_G[i])))
        ag_list.append(ag)
        i = i+1

    ag_list = np.array(ag_list)

    count_po_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list >= ag_space_G*i)
        count_po_list.append(count)
        i = i+1

    count_po_list = np.array(count_po_list)

    count_ne_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list <= ag_space_G*(-i))
        count_ne_list.append(count)
        i = i+1

    count_ne_list = np.array(count_ne_list)

    return ag_list, count_po_list, count_ne_list

def fit_Projection_vs_ag(para_1, para_2, x, y):

    def func(para_1, para_2, p, x):
        c = p
        return para_1 * ((para_2-x)/para_2)**c  #Grasselli

    # 定义误差函数
    def error(p, x, y):
        return func(para_1, para_2, p, x) - y

    # 初始参数
    p0 = [1]

    # 最小二乘法拟合
    para_fit, cov = leastsq(error, p0, args=(x, y))

    # 计算拟合优度值
    ss_err = ((func(para_1, para_2, para_fit, x)-y)** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared =1 - ss_err / ss_tot

    return para_fit, r_squared

def sampling_interval_vs_Grasselli(sampling_interval,ag_space,x,y):

    interval_space = sampling_interval/0.05 

    x1 = x[0::int(interval_space)]
    y1 = y[0::int(interval_space)]

    ag_list, count_po_list, count_ne_list= ag_cal(x1,y1,ag_space)

    ag_list_po = ag_list[ag_list > 0]
    ag_list_ne = np.absolute(ag_list[ag_list < 0])

    po_lg = count_po_list / (len(x1)-1)
    ne_lg = count_ne_list / (len(x1)-1)

    po_lg = po_lg[po_lg != 0] 
    ne_lg = ne_lg[ne_lg != 0] 

    ag_x_po = np.linspace(0,int(len(po_lg))*ag_space-1,int(len(po_lg)))
    ag_x_ne = np.linspace(0,int(len(ne_lg))*ag_space-1,int(len(ne_lg)))

    G_c_po, r2_po = fit_Projection_vs_ag(np.max(po_lg), np.max(ag_list_po), ag_x_po, po_lg)
    G_c_ne, r2_ne = fit_Projection_vs_ag(np.max(ne_lg), np.max(ag_list_ne), ag_x_ne, ne_lg)

    G_value_po = np.max(ag_list_po)/(G_c_po + 1)
    G_value_ne = np.max(ag_list_ne)/(G_c_ne + 1)

    return po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne

df = pd.read_excel('sandstone_profile.xlsx', sheet_name='sin', header=0, usecols='A:J', nrows=1001)
x = np.array(df.values[0:936,4])
y = np.array(df.values[0:936,5])

sampling_interval = 10
ag_space = 1

po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne = sampling_interval_vs_Grasselli(sampling_interval,ag_space,x,y)

print(f"forward  lg: {np.max(po_lg):.3f},", f"max ag:{np.max(ag_list_po):.3f},", f"c:{G_c_po[0]:.3f},", f"fitting: {r2_po:.3f},",f"G_value:{G_value_po[0]:.3f}")
print(f"backward lg: {np.max(ne_lg):.3f},", f"max ag:{np.max(ag_list_ne):.3f},", f"c:{G_c_ne[0]:.3f},", f"fitting: {r2_ne:.3f},",f"G_value:{G_value_ne[0]:.3f}")

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig2 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,90)
ax.set_ylim(0,0.7)
ax.set_xticks(np.arange(0,90,5))
ax.set_yticks(np.arange(0,0.8,0.1))
ax.set_ylabel('$A_0$')
ax.set_xlabel('ag of inclination/°')
ax.grid(True,ls='--')

# ax.bar(ag_x_po, po_lg)
ax.scatter(ag_x_po, po_lg,marker='o',color='red',s=50,label='forward test')
ax.scatter(ag_x_ne, ne_lg,marker='o',color='green',s=50,label='backward test')
ax.plot(ag_x_po, np.max(po_lg) * ((np.max(ag_list_po)-ag_x_po)/np.max(ag_list_po))**G_c_po, color='red', linestyle='solid', lw=2,label='forward fit')
ax.plot(ag_x_ne, np.max(ne_lg) * ((np.max(ag_list_ne)-ag_x_ne)/np.max(ag_list_ne))**G_c_ne, color='green', linestyle='solid', lw=2,label='backward fit')
ax.legend()

plt.savefig('Grasselli.svg')
plt.show()

# %%  计算不同sampling interval的Grasselli值
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

def ag_cal(x_G, y_G, ag_space_G):

    ag_list = []

    for i in range(len(x_G)-1):

        ag = np.rad2deg(np.arctan((y_G[i+1]-y_G[i])/(x_G[i+1]-x_G[i])))
        ag_list.append(ag)
        i = i+1

    ag_list = np.array(ag_list)

    count_po_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list >= ag_space_G*i)
        count_po_list.append(count)
        i = i+1

    count_po_list = np.array(count_po_list)

    count_ne_list = []

    for i in range(int(90/ag_space_G)):

        count = np.sum(ag_list <= ag_space_G*(-i))
        count_ne_list.append(count)
        i = i+1

    count_ne_list = np.array(count_ne_list)

    return ag_list, count_po_list, count_ne_list

def fit_Projection_vs_ag(para_1, para_2, x, y):

    def func(para_1, para_2, p, x):
        c = p
        return para_1 * ((para_2-x)/para_2)**c  #Grasselli

    # 定义误差函数
    def error(p, x, y):
        return func(para_1, para_2, p, x) - y

    # 初始参数
    p0 = [1]

    # 最小二乘法拟合
    para_fit, cov = leastsq(error, p0, args=(x, y))

    # 计算拟合优度值
    ss_err = ((func(para_1, para_2, para_fit, x)-y)** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared =1 - ss_err / ss_tot

    return para_fit, r_squared

def sampling_interval_vs_Grasselli(sampling_interval,ag_space,x,y):

    interval_space = sampling_interval/0.05 

    x1 = x[0::int(interval_space)]
    y1 = y[0::int(interval_space)]

    ag_list, count_po_list, count_ne_list= ag_cal(x1,y1,ag_space)

    ag_list_po = ag_list[ag_list > 0]
    ag_list_ne = np.absolute(ag_list[ag_list < 0])

    po_lg = count_po_list / (len(x1)-1)
    ne_lg = count_ne_list / (len(x1)-1)

    po_lg = po_lg[po_lg != 0] 
    ne_lg = ne_lg[ne_lg != 0] 

    ag_x_po = np.linspace(0,int(len(po_lg))*ag_space-1,int(len(po_lg)))
    ag_x_ne = np.linspace(0,int(len(ne_lg))*ag_space-1,int(len(ne_lg)))

    G_c_po, r2_po = fit_Projection_vs_ag(np.max(po_lg), np.max(ag_list_po), ag_x_po, po_lg)
    G_c_ne, r2_ne = fit_Projection_vs_ag(np.max(ne_lg), np.max(ag_list_ne), ag_x_ne, ne_lg)

    G_value_po = np.max(ag_list_po)/(G_c_po + 1)
    G_value_ne = np.max(ag_list_ne)/(G_c_ne + 1)

    return po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne

df = pd.read_excel('sandstone_profile.xlsx', sheet_name='10-2-yz', header=0, usecols='A:J', nrows=1001)
x = np.array(df.values[0:916,6])
y = np.array(df.values[0:916,7])

ag_space = 1

Inter = np.linspace(0.05,10,200)
G_value_po_list = []
G_value_ne_list = []

for i in range(len(Inter)):

    G_value_po = sampling_interval_vs_Grasselli(Inter[i], ag_space, x, y)[5]
    G_value_po_list.append(G_value_po[0])
    i = i + 1

G_value_po_list = np.array(G_value_po_list)

for i in range(len(Inter)):

    G_value_ne = sampling_interval_vs_Grasselli(Inter[i], ag_space, x, y)[11]
    G_value_ne_list.append(G_value_ne[0])
    i = i + 1

G_value_ne_list = np.array(G_value_ne_list)

# print(G_value_ne_list)

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig2 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,12)
ax.set_xticks(np.arange(0,11,5))
ax.set_yticks(np.arange(0,12,1))
ax.set_ylabel('G_value')
ax.set_xlabel('sampling interval')
ax.grid(True,ls='--')

ax.scatter(Inter, G_value_po_list, marker='o',color='red',s=50,label='po')
ax.scatter(Inter, G_value_ne_list, marker='s',color='green',s=50,label='ne')
# ax.plot(ag_x_po, np.max(po_lg) * ((np.max(ag_list_po)-ag_x_po)/np.max(ag_list_po))**G_c_po, color='red', linestyle='solid', lw=2,label='forward fit')
ax.legend()

plt.savefig('Grasselli.svg')
plt.show()