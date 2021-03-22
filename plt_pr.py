from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
figure, ax = plt.subplots()

#500
sh=500
if sh==500:
    pre11=[0.3139 , 0.7230 , 0.7451 , 0.7596 , 0.7676 , 0.7748 , 0.7823 , 0.7922 , 0.7979 , 0.8087]

    re11=[0.9980 , 0.7333 , 0.7105 , 0.6952 , 0.6848 , 0.6767 , 0.6691 , 0.6592 , 0.6477 , 0.6323]
else:
    pre11=[0.3012 , 0.7209 , 0.7441 , 0.7593 , 0.7700 , 0.7786 , 0.7869 , 0.7976 , 0.8090 , 0.8219]

    re11=[1.0000 , 0.9048 , 0.8937 , 0.8852 , 0.8791 , 0.8738 , 0.8686 , 0.8615 , 0.8526 , 0.8396]

plt.plot(re11,pre11,label='ours',linestyle='-',color='#006400',markersize = 10,linewidth = 3)

plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])  #设置x,y坐标值
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.tick_params(labelsize=20,which='major')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 22,
'FontWeight':'bold',
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}
plt.legend(prop=font2,loc='lower right',ncol=2,frameon=False)
# x_major_locator=MultipleLocator(2)
# ax.xaxis.set_major_locator(x_major_locator)
#borderaxespad=0,bbox_to_anchor=(1,0)
plt.grid(axis="x")
plt.grid(axis="y")
plt.xlabel('Recall',font1)
plt.ylabel('F-measure',font1)

ax = plt.gca()
# ax.set_aspect(9)
# plt.title('Line Chart',font1)
plt.tight_layout()
s='PR'+str(sh)+'.pdf'
plt.savefig(s)
plt.show()