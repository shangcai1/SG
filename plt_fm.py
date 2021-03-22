from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
figure, ax = plt.subplots()

#500
sh=100
if sh==500:
    fm11=[0.6331 , 0.6838 , 0.6867 , 0.6869 , 0.6868 , 0.6870 , 0.6874 , 0.6877 , 0.6872 , 0.6850 , 0]
else:
    fm11=[0.6702 , 0.7464 , 0.7634 , 0.7729 , 0.7800 , 0.7850 , 0.7903 , 0.7963 , 0.8026 , 0.8083 , 0]

x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

plt.plot(x,fm11,label='ours',linestyle='-',color='#006400',markersize = 10,linewidth = 3)

plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks(x)
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
# plt.legend(prop=font1,loc=[0.1,1.05],ncol=4,frameon=False)
plt.legend(prop=font2,loc='lower left',ncol=2,frameon=False)
# x_major_locator=MultipleLocator(2)
# ax.xaxis.set_major_locator(x_major_locator)
#borderaxespad=0,bbox_to_anchor=(1,0)
plt.grid(axis="x")
plt.grid(axis="y")
plt.xlabel('Threshold',font1)
plt.ylabel('F-measure',font1)

ax = plt.gca()
# ax.set_aspect(9)
# plt.title('Line Chart',font1)
plt.tight_layout()
s='fm'+str(sh)+'.pdf'
plt.savefig(s)
plt.show()
# pp.savefig()
# pp.close()