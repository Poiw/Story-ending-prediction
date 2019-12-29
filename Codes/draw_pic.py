import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('../Models/tmp/loss.txt') as f:
    buf = f.readlines()

loss = []
acc = []
epoch = []

for i, line in enumerate(buf):
    epoch.append(i)
    line = line.split()
    loss.append(float(line[0]))
    acc.append(float(line[1]))

# plt.plot(epoch,loss,label='Training Loss',color='r')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('../Models/tmp/result.pdf')
plt.plot(epoch,acc,label='Validation Accuracy',color='b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../Models/tmp/result2.pdf')