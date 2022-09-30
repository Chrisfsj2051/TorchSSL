import matplotlib.pyplot as plt
import numpy

# warmup, confidence
x = [0.6, 0.7, 0.75, 0.8, 0.90, 0.95, 0.99]
vcc_y = [5.92, 5.63, 5.42, 5.55, 5.90, 6.56, 7.08]
flexmatch_y = [7.21, 6.81, 6.40, 6.28, 6.10, 5.95, 6.21]
plt.plot(x, flexmatch_y, marker='^')
plt.plot(x, vcc_y, marker='o')
plt.xlabel('Threshold τ', fontdict={'size': 16})
plt.ylabel('Error rate (%)', fontdict={'size': 15})
plt.xticks([0.6, 0.7, 0.75, 0.8, 0.90, 0.95, 0.99],
           ['0.6', '0.7', '0.75', '0.8', '0.9', '0.95', '0.99'], size=14)
plt.yticks(size=14)
plt.legend(['FlexMatch', 'VCC'], loc="upper center", prop={'size': 13})
plt.grid()
plt.savefig('./exp1.pdf', bbox_inches='tight')
plt.cla()

x = [1, 2, 3]
y = [11.43, 7.82, 5.42]
plt.bar(x, y, color=['#336699', '#FFCC33', '#339933'], )
plt.ylabel('Error rate (%)', fontdict={'size': 13})
plt.xticks([1, 2, 3], ['without Stage1', 'without Stage2', 'Full warm-up\nstrategy'], size=13)
plt.yticks(range(0, 13, 2), [f'{i}.0' for i in range(0, 13, 2)])
plt.savefig('./exp2.pdf', bbox_inches='tight')
# plt.show()
