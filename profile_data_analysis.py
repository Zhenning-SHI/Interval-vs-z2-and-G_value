# %%  绘制轮廓线 *********************************
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

input_data = ['06x', 1, 0.05, 1000, 2]

# region
df = pd.read_excel('profile_data.xlsx', 
                   sheet_name = input_data[0], 
                   header = 0, 
                   usecols = 'A:E', 
                   nrows = 1001)
x = df.values[0: input_data[3], 0]
y = df.values[0: input_data[3], input_data[1]]

sampling_interval = input_data[2]
interval_space = sampling_interval / 0.05

x1 = x[0: : int(interval_space)]
y1 = y[0: : int(interval_space)]

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '24'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig1 = plt.figure(figsize = (24, 3), 
                  edgecolor = 'white', 
                  dpi = 200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1, 51)
ax.set_ylim(-3, 3)
ax.set_xticks(np.arange(0, 55, 5))
ax.set_yticks(np.arange(-3, 3, 1))
ax.set_ylabel('y/mm')
ax.set_xlabel('x/mm')
ax.grid(True,
        ls = '--')
ax.scatter(x1, y1,
           marker = '^',
           color = 'blue',
           s = 10,
           label = 'profile')
ax.plot(x1, y1,
        color = 'blue',
        lw = 2,
        label = 'profile')

plt.savefig('./fig/profile/profile_' +
            str(input_data[0]) + '_' +
            str(input_data[1]) + '_' +
            str(input_data[2]) + '.pdf')

plt.show()
# endregion
# %%  计算Grasselli值
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

input_data = ['06x', 5, 6.3, int(1), 1000]

# region
def ag_cal(x_G, y_G, ag_space_G):

    ag_list = []

    for i in range(len(x_G) - 1):

        ag = np.rad2deg(np.arctan((y_G[i + 1] - y_G[i]) / (x_G[i + 1] - x_G[i])))
        ag_list.append(ag)

        i = i + 1

    ag_list = np.array(ag_list)

    count_po_list = []

    for i in range(int(90 / ag_space_G)):

        count = np.sum(ag_list >= ag_space_G*i)
        count_po_list.append(count)

        i = i + 1

    count_po_list = np.array(count_po_list)

    count_ne_list = []

    for i in range(int(90 / ag_space_G)):

        count = np.sum(ag_list <= ag_space_G * (-i))
        count_ne_list.append(count)
        i = i + 1

    count_ne_list = np.array(count_ne_list)

    return ag_list, count_po_list, count_ne_list

def fit_Projection_vs_ag(para_1, para_2, x, y):

    def func(para_1, para_2, p, x):
        c = p
        return para_1 * ((para_2 - x) / para_2) ** c  #Grasselli

    # 定义误差函数
    def error(p, x, y):
        return func(para_1, para_2, p, x) - y

    # 初始参数
    p0 = [1]

    # 最小二乘法拟合
    para_fit, cov = leastsq(error, p0, args=(x, y))

    # 计算拟合优度值
    ss_err = ((func(para_1, para_2, para_fit, x) - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared =1 - ss_err / ss_tot

    return para_fit, r_squared

def SI_vs_Gvalue(sampling_interval, ag_space, x, y):

    interval_space = sampling_interval / 0.05 

    x1 = x[0: : int(interval_space)]
    y1 = y[0: : int(interval_space)]

    ag_list, count_po_list, count_ne_list = ag_cal(x1, y1, ag_space)

    ag_list_po = ag_list[ag_list > 0]
    ag_list_ne = np.absolute(ag_list[ag_list < 0])

    po_lg = count_po_list / (len(x1) - 1)
    ne_lg = count_ne_list / (len(x1) - 1)

    po_lg = po_lg[po_lg != 0] 
    ne_lg = ne_lg[ne_lg != 0] 

    ag_x_po = np.linspace(0, int(len(po_lg)) * ag_space - 1,int(len(po_lg)))
    ag_x_ne = np.linspace(0, int(len(ne_lg)) * ag_space - 1, int(len(ne_lg)))

    G_c_po, r2_po = fit_Projection_vs_ag(np.max(po_lg), np.max(ag_list_po), ag_x_po, po_lg)
    G_c_ne, r2_ne = fit_Projection_vs_ag(np.max(ne_lg), np.max(ag_list_ne), ag_x_ne, ne_lg)
    
    G_value_po = np.max(ag_list_po) / (G_c_po + 1)
    G_value_ne = np.max(ag_list_ne) / (G_c_ne + 1)

    return po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne

df = pd.read_excel('profile_data.xlsx', 
                   sheet_name = input_data[0], 
                   header = 0, 
                   usecols = 'A:F', 
                   nrows = 1001)
x = np.array(df.values[0: input_data[4], 0])
y = np.array(df.values[0: input_data[4], input_data[1]])

sampling_interval = input_data[2]
ag_space = input_data[3]

po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne = SI_vs_Gvalue(sampling_interval, ag_space, x, y)

# print(f"forward  lg: {np.max(po_lg):.3f},", f"max ag:{np.max(ag_list_po):.3f},", f"c:{G_c_po[0]:.3f},", f"fitting: {r2_po:.3f},",f"G_value:{G_value_po[0]:.3f}")
# print(f"backward lg: {np.max(ne_lg):.3f},", f"max ag:{np.max(ag_list_ne):.3f},", f"c:{G_c_ne[0]:.3f},", f"fitting: {r2_ne:.3f},",f"G_value:{G_value_ne[0]:.3f}")

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
# plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1, 90)
ax.set_ylim(0, 0.7)
ax.set_xticks(np.arange(0, 91, 5))
ax.set_yticks(np.arange(0, 0.8, 0.1))
ax.set_ylabel('$\mathregular{A_0}$')
ax.set_xlabel('Angle of inclination/°')
ax.grid(True, 
        ls='--')

ax.bar(ag_x_po, po_lg, 
       color = 'red', 
       alpha = 0.5)
ax.bar(ag_x_ne, ne_lg, 
       color = 'green', 
       alpha = 0.5)
ax.scatter(ag_x_po, po_lg,
           marker = 'o',
           color = 'red',
           s = 50,
           label = 'forward test')
ax.scatter(ag_x_ne, ne_lg,
           marker = 'o',
           color = 'green',
           s = 50,
           label = 'backward test')
ax.plot(ag_x_po, np.max(po_lg) * ((np.max(ag_list_po) - ag_x_po) / np.max(ag_list_po)) ** G_c_po, 
        color = 'red', 
        linestyle = 'solid', 
        lw = 2,
        label = 'forward fit')
ax.plot(ag_x_ne, np.max(ne_lg) * ((np.max(ag_list_ne) - ag_x_ne) / np.max(ag_list_ne)) ** G_c_ne, 
        color = 'green', 
        linestyle = 'solid', 
        lw = 2,
        label = 'backward fit')
ax.legend(loc=(65/90,0.3/0.7))

ax.text(25, 0.53,
        f'Specimen No.{input_data[0]} Profile No.{input_data[1]} Interval:{input_data[2]} Ag_spcae:{input_data[3]}')
ax.text(40, 0.01,
        f'Forward \nlg:{np.max(po_lg):.3f} \nAg_max:{np.max(ag_list_po):.3f} \nC_value:{G_c_po[0]:.3f}'+'\n$\mathregular{\\theta_{max}^{*}/(C+1)_{2D}}$:'+f'{G_value_po[0]:.3f}'+'\n$\mathregular{R^2}$:'+f'{r2_po:.3f}')
ax.text(65, 0.01,
        f'Backward \nlg:{np.max(ne_lg):.3f} \nAg_max:{np.max(ag_list_ne):.3f} \nC_value:{G_c_ne[0]:.3f}'+'\n$\mathregular{\\theta_{max}^{*}/(C+1)_{2D}}$:'+f'{G_value_ne[0]:.3f}'+'\n$\mathregular{R^2}$:'+f'{r2_ne:.3f}')

pfPlot = fig.add_axes([0.2, 0.78, 0.65, 0.08],
                      facecolor = 'lightyellow')
pfPlot.tick_params(labelsize = 12)
pfPlot.set_xlim(-1, 51)
pfPlot.set_ylim(-3, 3)
pfPlot.set_xlabel('x/mm', 
                  fontsize=12)
pfPlot.set_ylabel('y/mm', 
                  fontsize=12)
pfPlot.scatter(x, y,
               marker = '^',
               color = 'blue',
               s = 2,
               label = 'profile')
pfPlot.plot(x, y,
            color = 'blue',
            lw = 1,
            label = 'profile')

plt.savefig('./fig/Grasselli/A0/A0_' +
            str(input_data[0]) + '_' +
            str(input_data[1]) + '_' +
            str(input_data[2]) + '.pdf')
plt.show()
# endregion
# %%  计算不同SI的Grasselli值
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

def fit_A0_vs_ag(para_1, para_2, x, y):

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

def SI_vs_Gvalue(sampling_interval,ag_space,x,y):

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

    G_c_po, r2_po = fit_A0_vs_ag(np.max(po_lg), np.max(ag_list_po), ag_x_po, po_lg)
    G_c_ne, r2_ne = fit_A0_vs_ag(np.max(ne_lg), np.max(ag_list_ne), ag_x_ne, ne_lg)

    G_value_po = np.max(ag_list_po)/(G_c_po + 1)
    G_value_ne = np.max(ag_list_ne)/(G_c_ne + 1)

    return po_lg, ag_x_po, ag_list_po, G_c_po, r2_po, G_value_po, ne_lg, ag_x_ne, ag_list_ne, G_c_ne, r2_ne, G_value_ne

def fit_Gvalue_vs_SI(x_list,y_list):

    def func(p, x):
        a, b = p
        # return a + b * x #linear
        return a * x + b

    # 定义误差函数
    def error(p, x, y):
        return func(p, x) - y

    # 初始参数
    p0 = [0, 0]

    # 最小二乘法拟合
    para, cov = leastsq(error, p0, args=(x_list, y_list))

    # 计算拟合优度值
    ss_err = ((func(para,x_list)-y_list.mean())** 2).sum()
    ss_tot = ((y_list - y_list.mean()) ** 2).sum()
    r_squared = ss_err / ss_tot

    return para, r_squared

input_data = ['06y', 5, int(1), 1001]

df = pd.read_excel('profile_data.xlsx', 
                   sheet_name=input_data[0], 
                   header=0, 
                   usecols='A:F', 
                   nrows=1001)

x = np.array(df.values[0:input_data[3],0])
y = np.array(df.values[0:input_data[3],input_data[1]])

ag_space = input_data[2]

Inter = np.linspace(0.05,10,200)
GvaluePoList = []
GvalueNeList = []
Inter_real = []

for i in range(len(Inter)):
    
    try:
        G_value_po = SI_vs_Gvalue(Inter[i], ag_space, x, y)[5]
    except:
        # G_value_po=[0]
        continue
    GvaluePoList.append(G_value_po[0])
    Inter_real.append(Inter[i])
    i = i + 1

GvaluePoList = np.array(GvaluePoList)
Inter_real = np.array(Inter_real)

for i in range(len(Inter)):
    
    try:
        G_value_ne = SI_vs_Gvalue(Inter[i], ag_space, x, y)[11]
    except:
        # G_value_ne = [0]
        continue
    GvalueNeList.append(G_value_ne[0])

GvalueNeList = np.array(GvalueNeList)

Gvalue_po_fitpara = fit_Gvalue_vs_SI(Inter_real,GvaluePoList)[0]
Gvalue_ne_fitpara = fit_Gvalue_vs_SI(Inter_real,GvalueNeList)[0]

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), 
                 edgecolor='white', 
                 dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1,11)
ax.set_ylim(0,12)
ax.set_xticks(np.arange(0,11,1))
ax.set_yticks(np.arange(0,13,1))
ax.set_ylabel('G_value')
ax.set_xlabel('Interval/mm')
ax.grid(True,ls='--')

ax.scatter(Inter_real, GvaluePoList, 
           marker='o',
           color='red',
           s=50,
           label='po'
           )
ax.scatter(Inter_real, GvalueNeList, 
           marker='s',
           color='green',
           s=50,
           label='ne')
ax.plot(Inter_real, 
        Gvalue_po_fitpara[0] * Inter_real + Gvalue_po_fitpara[1], 
        color='red',
        lw=4)
ax.plot(Inter_real, 
        Gvalue_ne_fitpara[0] * Inter_real + Gvalue_ne_fitpara[1], 
        color='green',
        lw=4)
ax.legend(loc=(9/12,7/12))
ax.text(0,1,f'Specimen No.{input_data[0]} \nProfile No.{input_data[1]} \nAg_space:{input_data[2]}')

pfPlot = fig.add_axes([.2, .78, .65, .08],facecolor='lightyellow')
pfPlot.tick_params(labelsize=12)
pfPlot.set_xlim(-1,51)
pfPlot.set_ylim(-3,3)
pfPlot.set_xlabel('x/mm', 
                  fontsize=12)
pfPlot.set_ylabel('y/mm', 
                  fontsize=12)
pfPlot.scatter(x,y,
               marker='^',
               color='blue',
               s=2,
               label='profile')
pfPlot.plot(x,y,
            color='blue',
            lw=1,
            label='profile')

plt.savefig('./fig/Grasselli/Gvalue/Gvalue_'+
            str(input_data[0])+'_'+
            str(input_data[1])+'_'+
            str(input_data[2])+'.pdf')
plt.show()
# %%  计算不同SI的Z2值
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import leastsq

input_data = ['06x', 1, int(200), 1000, 2]

# region

def z2_cal(interval, x, y):

    space = interval/0.05
    k_arr = []
    x1 = x[0: : int(space)]
    y1 = y[0: : int(space)]

    for i in range(len(x1) - 1):

        k = ((y1[i + 1] - y1[i]) / (x1[i + 1] - x1[i]))
        k_arr.append(k)

        i = i + 1

    k_arr = np.array([k_arr]) 

    k_arr_po = k_arr[k_arr > 0]
    k_arr_ne = k_arr[k_arr < 0]

    z_2 = np.average(np.square(k_arr)) ** 0.5
    z_2_po = np.average(np.square(k_arr_po)) ** 0.5
    z_2_ne = np.average(np.square(k_arr_ne)) ** 0.5

    return z_2, z_2_po, z_2_ne

def fit_z2_vs_interval(Iv, z2_list):

    def func(p, x):
        a, b, c = p
        # return a + b * x #linear
        return a * x ** b + c

    # 定义误差函数
    def error(p, x, y):
        return func(p, x) - y

    # 初始参数
    p0 = [1, 1, 1]

    # 最小二乘法拟合
    para, cov = leastsq(error, p0, args=(Iv, z2_list))

    # 计算拟合优度值
    ss_err = ((func(para,Iv) - z2_list.mean() ) ** 2).sum()
    ss_tot = ((z2_list - z2_list.mean()) ** 2).sum()
    r_squared = ss_err / ss_tot

    return para, r_squared

df = pd.read_excel('profile_data.xlsx',
                    sheet_name = input_data[0],
                    header = 0, 
                    usecols = 'A:F', 
                    nrows = 1001)
x = np.array(df.values[0: input_data[3], 0])
y = np.array(df.values[0: input_data[3], input_data[1]])

Inter = np.linspace(0.05, 10, input_data[2])
z2_list = []
z2_list_po = []
z2_list_ne = []

for i in range(len(Inter)):

    z2 = z2_cal(Inter[i], x, y)[0]
    z2_list.append(z2)

    z2_po = z2_cal(Inter[i], x, y)[1]
    z2_list_po.append(z2_po)

    z2_ne = z2_cal(Inter[i], x, y)[2]
    z2_list_ne.append(z2_ne)

    i = i + 1

z2_list = np.array(z2_list)
z2_list_po = np.array(z2_list_po)
z2_list_ne = np.array(z2_list_ne)

para, r_squared = fit_z2_vs_interval(Inter, z2_list)
paraPo, r_squared = fit_z2_vs_interval(Inter, z2_list_po)
paraNe, r_squared = fit_z2_vs_interval(Inter, z2_list_ne)

plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
ax = plt.subplot(111)

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.set_xlim(-1, 11)
ax.set_ylim(0, 0.3)
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 0.31, 0.02))
ax.set_ylabel('$\mathregular{Z_2}$/mm')
ax.set_xlabel('Interval/mm')
ax.grid(True, ls = '--')

if input_data[4] == 1:
    ax.scatter(Inter, z2_list, 
            marker = '^', 
            color = 'blue', 
            s = 50,
            label = '$\mathregular{Z_2}$')
    ax.plot(Inter, para[0] * Inter ** para[1] + para[2], 
            color = 'red', 
            linestyle = 'solid', 
            lw = 2,
            label = 'fitting line')
    
else:
    ax.scatter(Inter, z2_list_po, 
            marker = 'o', 
            color = 'red', 
            s = 50,
            label = '$\mathregular{Z_2}$ forward')
    ax.plot(Inter, paraPo[0] * Inter ** paraPo[1] + paraPo[2], 
            color = 'red', 
            linestyle = 'solid', 
            lw = 2,
            label = 'fitting line Po')
    ax.scatter(Inter, z2_list_ne, 
            marker = 's', 
            color = 'green', 
            s = 50,
            label = '$\mathregular{Z_2}$ backward')
    ax.plot(Inter, paraNe[0] * Inter ** paraNe[1] + paraNe[2], 
            color = 'green', 
            linestyle = 'solid', 
            lw = 2,
            label = 'fitting line Ne')
    
ax.legend(loc=(8/12,16/30))
ax.text(0,0.02,
        f'Specimen No.{input_data[0]} \nProfile No.{input_data[1]} \nSI:{input_data[2]} \nA={para[0]:.3f} B={para[1]:.3f} C={para[2]:.3f}' +
        '\n$\mathregular{R^2}$=' +
        f'{r_squared:.3f}' +
        f'\n z2max={np.max(z2_list):.3f} z2min={np.min(z2_list):.3f}')

pfPlot = fig.add_axes([0.2, 0.78, 0.65, 0.08],
                      facecolor = 'lightyellow')
pfPlot.tick_params(labelsize = 12)
pfPlot.set_xlim(-1, 51)
pfPlot.set_ylim(-3, 3)
pfPlot.set_xlabel('x/mm', 
                  fontsize = 12)
pfPlot.set_ylabel('y/mm', 
                  fontsize = 12)
pfPlot.scatter(x, y,
               marker = '^',
               color = 'blue',
               s = 2,
               label = 'profile')
pfPlot.plot(x, y,
            color = 'blue',
            lw = 1,
            label = 'profile')

plt.savefig('./fig/Z2/Z2_' +
            str(input_data[0]) + '_' +
            str(input_data[1]) + '_' +
            str(input_data[2]) + '.pdf')

plt.show()

# endregion
# %%  角度分布直方图并进行正态检验
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as ss

input_data = ['sin', 1, 0.05, 1001]

# region
def ag_cal(x, y):

    ag_list = []

    for i in range(len(x) - 1):

        ag = np.rad2deg(np.arctan((y[i + 1] - y[i])/(x[i + 1] - x[i])))
        ag_list.append(ag)
        i = i + 1

    ag_list = np.array(ag_list)

    return ag_list

df = pd.read_excel('profile_data.xlsx', 
                   sheet_name=input_data[0], 
                   header=0, 
                   usecols='A:F', 
                   nrows=1001)
x = np.array(df.values[0: input_data[3], 0])
y = np.array(df.values[0: input_data[3], input_data[1]])

sampling_interval = input_data[2]
interval_space = sampling_interval / 0.05

x1 = x[0: : int(interval_space)]
y1 = y[0: : int(interval_space)]

angle_list = ag_cal(x1, y1) 

angle_list_mean = np.mean(angle_list)
angle_list_var = np.var(angle_list)
angle_list_std = np.std(angle_list)

# test
ks_test = ss.kstest(angle_list, 
                    'norm', 
                    args = (angle_list_mean, angle_list_std))
ad_test = ss.anderson(angle_list, dist = 'norm')
sw_test = ss.shapiro(angle_list)

# print(f"mean angle:{angle_list_mean:.3f}", f"angle var:{angle_list_std:.3f}", ks_test,ad_test,sw_test)
# print(angle_list)

# style
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# distribution
fig1 = plt.figure(figsize = (12,8), 
                  edgecolor = 'white', 
                  dpi = 200)
ax1 = plt.subplot(111)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, 0.1)
ax1.set_xticks(np.arange(-90, 91, 10))
ax1.set_yticks(np.arange(0, 0.11, 0.01))
ax1.set_ylabel('Probability Density')
ax1.set_xlabel('Angle of inclination/°')
ax1.grid(True, ls = '--')
ax1.hist(angle_list, 
         range = (-90, 90), 
         bins = 180,
         density = True, 
         alpha = 1, 
         color = 'red', 
         rwidth = 0.5)
ax1.plot(np.linspace(-90, 90, 180), np.exp(-(np.linspace(-90, 90, 180) - angle_list_mean) ** 2 / (2 * angle_list_std ** 2)) / (np.sqrt(2 * np.pi) * angle_list_std), 
         color = 'green', 
         linestyle = 'solid', 
         lw = 4,
         label = 'forward fit')
ax1.text(40, 0.045, 
         f'Quartzite No.{input_data[0]} \nProfile No.{input_data[1]} \nSI:{input_data[2]}mm \nMean={angle_list_mean:.3f} \nStd={angle_list_std:.3f} \nP_value={ks_test[1]:.3f}')

pfPlot = fig1.add_axes([0.2, 0.78, 0.65, 0.08],
                       facecolor = 'lightyellow')
pfPlot.tick_params(labelsize = 12)
pfPlot.set_xlim(-1, 51)
pfPlot.set_ylim(-3, 3)
pfPlot.set_xlabel('x/mm', fontsize = 12)
pfPlot.set_ylabel('y/mm', fontsize = 12)
pfPlot.scatter(x1, y1,
               marker = '^',
               color = 'blue',
               s = 2,
               label = 'profile')
pfPlot.plot(x1,y1,
            color = 'blue',
            lw = 1,
            label = 'profile')

sorted_ = np.sort(angle_list)
yvals = np.arange(len(sorted_)) / float(len(sorted_))
x_label = ss.norm.ppf(yvals)

qqplot = fig1.add_axes([0.2, 0.4, 0.2, 0.3], 
                       facecolor = 'lightcyan')
qqplot.tick_params(labelsize = 12)
qqplot.set_xlabel('Theoretical Quantiles', 
                  fontsize = 12)
qqplot.set_ylabel('Sample Quantiles', 
                  fontsize = 12)
qqplot.set_title('q-q plot', 
                 fontsize = 12)
qqplot.scatter(x_label, sorted_, 
               marker = '^', 
               color = 'blueviolet', 
               s = 4,
               label = 'profile')

plt.savefig('./fig/distribution/SI/dist_' +
            str(input_data[0]) + '_' +
            str(input_data[1]) + '_' +
            str(input_data[2]) + '.pdf')

# qqplot = fig1.add_axes([.2, .5, .2, .2])
# qqplot.tick_params(labelsize=10)
# ss.probplot(angle_list, dist='norm', plot=qqplot)

# qqplot
# fig2 = plt.figure(figsize=(12,8), edgecolor='white', dpi=200)
# ax2 = plt.subplot(111)
# qqplot = ss.probplot(angle_list, dist='norm', plot=ax2)
# plt.savefig('qq_'+str(input_data[0])+'_'+str(input_data[1])+'_'+str(input_data[2])+'.svg')

plt.show()

# endregion
# %%  绘制玫瑰图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

# get data
N = 4
theta = np.linspace(0.0, 2 * np.pi, N, endpoint = False)
theta = np.append(theta, theta[0])
radii = [8.569, 7.148, 6.723, 8.152]
radii = np.append(radii, radii[0])

# style
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(8,8), 
                 edgecolor='white', 
                 dpi=200)
ax = plt.subplot(projection = 'polar')
# ax.set_theta_offset(0)
ax.set_rlabel_position(0)
ax.set_ylim(0, 10)
ax.set_yticks(np.arange(0, 10, 2))
ax.set_ylabel('z2')
ax.yaxis.set_label_coords(0.48, 0.75)
# ax.plot(theta, radii, color='red', lw=2)
# ax.fill(theta, radii, facecolor='blue', alpha=0.7)
color_s = cm.Reds(radii / np.max(radii))
ax.bar(theta, radii, 
       width=(2 * np.pi / N), 
       bottom = 0.0, 
       color = color_s, 
       alpha = 1)
norm = colors.Normalize(0, np.ceil(np.max(radii)))
fig.colorbar(cm.ScalarMappable(norm = norm, cmap = 'Reds'), ax = ax, fraction = 0.045, pad = 0.07)

plt.show()
# %%  绘制表格
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plottable import Table

# d = pd.DataFrame(np.random.random((10, 5)), columns=["A", "B", "C", "D", "E"]).round(2)
df = pd.read_excel('profile_data.xlsx', 
                   header=0,
                   sheet_name='Grasselli',
                   usecols='A:E')
df=pd.DataFrame(df)
df = df.astype(float).round(2)
print(df)
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '20'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

fig = plt.figure(figsize=(12,8), 
                 edgecolor='white', 
                 dpi=200)
ax = plt.subplot(111)
tab = Table(df,footer_divider = True,
            row_dividers = True,
            index_col = None)
plt.savefig('table.pdf')
plt.show()