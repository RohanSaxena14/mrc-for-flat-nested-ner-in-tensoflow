from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

host = host_subplot(111)
par = host.twinx()

host.set_xlabel("epochs")
host.set_ylabel("accuracy")

par.set_ylabel("loss")

print(history.history)

with open('history.json', 'w') as fp:
    json.dump(history.history, fp)

p1, = host.plot(history.history['start_flat_accuracy'], label="start_flat_accuracy")
p2, = host.plot(history.history['val_start_flat_accuracy'],label = 'val_start_flat_accuracy')
p3, = par.plot(history.history['start_flat_loss'], label="start_flat_loss")
p4, = par.plot(history.history['val_start_flat_loss'],label = "val_start_flat_loss")
leg = plt.legend()

plt.show()

host = host_subplot(111)
par = host.twinx()

host.set_xlabel("epochs")
host.set_ylabel("accuracy")

par.set_ylabel("loss")

p1, = host.plot(history.history['end_flat_accuracy'], label="end_flat_accuracy")
p2, = host.plot(history.history['val_end_flat_accuracy'],label = 'val_end_flat_accuracy')
p3, = par.plot(history.history['end_flat_loss'], label="end_flat_loss")
p4, = par.plot(history.history['val_end_flat_loss'],label = "val_end_flat_loss")
leg = plt.legend()

plt.show()

host = host_subplot(111)
par = host.twinx()

host.set_xlabel("epochs")
host.set_ylabel("accuracy")

par.set_ylabel("loss")

p1, = host.plot(history.history['span_flat_accuracy'], label="span_flat_accuracy")
p2, = host.plot(history.history['val_span_flat_accuracy'],label = 'val_span_flat_accuracy')
p3, = par.plot(history.history['span_flat_loss'], label="span_flat_loss")
p4, = par.plot(history.history['val_span_flat_loss'],label = "val_span_flat_loss")
leg = plt.legend()

plt.show()