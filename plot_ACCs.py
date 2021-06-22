import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



epoch_total = 200
path = './results/OCTAMax_cycle_gan_cnn_1_5_00006_64'
filename = '%s/ACCs.data' % path # txt文件和当前脚本在同一目录下，所以不用写具体路径
epoch = []
ACCs_A = []
ACCs_B = []
ACCs = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
            pass
        epoch_tmp, ACC_A_tmp, ACC_B_tmp, ACC_tmp = [float(i) for i in lines.split('\t')] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        epoch.append(epoch_tmp)  # 添加新读取的数据
        ACCs_A.append(ACC_A_tmp)
        ACCs_B.append(ACC_B_tmp)
        ACCs.append(ACC_tmp)
        pass
    epoch = np.array(epoch) # 将数据从list类型转换为array类型。
    ACCs_A = np.array(ACCs_A)
    ACCs_B = np.array(ACCs_B)
    ACCs = np.array(ACCs)
    pass

x_axis_data = range(5, epoch_total + 1, 5)
plt_A, = plt.plot(x_axis_data, ACCs_A)
plt_B, = plt.plot(x_axis_data, ACCs_B)
plt_, = plt.plot(x_axis_data, ACCs)
plt.legend(handles=[plt_A, plt_B, plt_], labels=['A', 'B', 'fusion'])
plt.xlabel('epoch')
plt.ylabel('ACCs')

plt.savefig("%s/ACCs.jpg" % (path))