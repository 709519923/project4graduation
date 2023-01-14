
#导入包
import matplotlib.pyplot as plt
import numpy as np

#创建数据
theta = np.linspace(0, np.pi, 100)
# unit   = 0.25
# theta = np.arange(0, 1+unit, unit)
rho1 = np.cos(theta)
rho2 = np.sin(theta)+2*np.cos(theta)
rho3 = 2*np.sin(theta)+3*np.cos(theta)
#创建figure窗口
plt.figure(num=3, figsize=(8, 5))
#画曲线1
plt.plot(theta, rho1)
#画曲线2
plt.plot(theta, rho2, color='blue')
plt.plot(theta, rho3, color='red')
#设置坐标轴范围
plt.xlim((0, np.pi))
plt.ylim((-5, 5))
#设置坐标轴名称
plt.xlabel('')
plt.ylabel('')
#设置坐标轴刻度
# my_x_ticks = np.arange(0, np.pi, np.pi/180)
# my_y_ticks = np.arange(-2, 2, 0.3)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)

x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$" , r"$\frac{3\pi}{4}$",   r"${\pi}$"]
plt.xticks(np.arange(0, 1+unit, unit)*np.pi , x_label, fontsize=16)

#显示出所有设置
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

x=np.arange(-10.0,10.0,0.1)
y=np.arctan(x)

fig = plt.figure()
ax  = fig.add_subplot(111)

ax.plot(x,y,'b.')

y_pi   = y/np.pi
unit   = 0.25
y_tick = np.arange(-0.5, 0.5+unit, unit)

y_label = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$+\frac{\pi}{4}$",   r"$+\frac{\pi}{2}$"]
ax.set_yticks(y_tick*np.pi)
ax.set_yticklabels(y_label, fontsize=15)

# y_label2 = [r"$" + format(r, ".2g")+ r"\pi$" for r in y_tick]
# ax2 = ax.twinx()
# ax2.set_yticks(y_tick*np.pi)
# ax2.set_yticklabels(y_label2, fontsize=20)

plt.show()