import matplotlib.pyplot as plt
import numpy as np

perceptron = {0.1: [(0.802, 1.5819759368896484), (0.809, 0.8072047233581543), (0.806, 0.7227659225463867),
       (0.795, 0.6132950782775879), (0.796, 0.6914889812469482)],
 0.2: [(0.823, 2.49029803276062), (0.814, 1.153473138809204), (0.828, 1.1118559837341309), (0.825, 1.0454912185668945),
       (0.831, 1.2424280643463135)],
 0.3: [(0.846, 1.5867009162902832), (0.842, 1.5739340782165527), (0.843, 1.4914522171020508),
       (0.846, 1.5944230556488037), (0.831, 1.437650203704834)],
 0.4: [(0.841, 3.740200996398926), (0.854, 9.135272026062012), (0.833, 1.855525016784668), (0.841, 1.6805918216705322),
       (0.844, 1.3547163009643555)],
 0.5: [(0.85, 2.122175931930542), (0.844, 1.8504996299743652), (0.838, 2.101778030395508), (0.837, 2.0513968467712402),
       (0.845, 2.171191692352295)],
 0.6: [(0.852, 2.405595302581787), (0.841, 2.5978591442108154), (0.836, 2.530411958694458), (0.837, 2.737779378890991),
       (0.843, 2.4633078575134277)],
 0.7: [(0.831, 2.6768579483032227), (0.846, 3.1585309505462646), (0.844, 2.8064236640930176),
       (0.837, 3.134660005569458), (0.843, 3.0164144039154053)],
 0.8: [(0.839, 3.0100090503692627), (0.846, 3.1087188720703125), (0.828, 3.2480273246765137),
       (0.841, 3.073374032974243), (0.842, 3.059084177017212)],
 0.9: [(0.838, 3.612950086593628), (0.844, 3.414395809173584), (0.833, 3.116532802581787), (0.84, 2.5035641193389893),
       (0.838, 3.3327300548553467)],
 1.0: [(0.837, 3.7805051803588867), (0.837, 3.584808826446533), (0.837, 3.661787986755371), (0.837, 3.4545650482177734),
       (0.837, 3.4595768451690674)]}


def plot_graph(data, name , train_size = None):
    x_axis = []
    y_axis_accuracy = []
    y_axis_training_time = []
    y_axis_standard_deviation = []
    for key in data:
        x_axis.append(key * 100)
        #   average
        accuracy = []
        training_time = []
        for acc, time in data[key]:
            accuracy.append(acc)
            training_time.append(time)
        y_axis_accuracy.append(np.average(accuracy) * 100)
        y_axis_training_time.append(np.average(training_time) * 100)
        y_axis_standard_deviation.append(np.std(accuracy))

    print(y_axis_accuracy)
    print(y_axis_training_time)
    print(y_axis_standard_deviation)

    plt.plot(x_axis, y_axis_accuracy)
    plt.ylabel(f'{name} Mean Accuracy ')
    plt.xlabel(f'Data %')
    plt.show()

    plt.plot(x_axis, y_axis_training_time)
    plt.ylabel(f'{name} Average  Training  Time in seconds')
    plt.xlabel(f'Data %')
    plt.show()

    plt.plot(x_axis, y_axis_standard_deviation)
    plt.ylabel(f'{name} Accuracy Standard Deviation')
    plt.xlabel(f'Data %')
    plt.show()

plot_graph(perceptron,'perceptron' )