import matplotlib.pyplot as plt
import numpy as np

perceptron = {0.1: [(0.6266666666666667, 0.405087947845459), (0.7266666666666667, 0.4866039752960205),
                    (0.6066666666666667, 0.26888394355773926), (0.6733333333333333, 0.5633711814880371),
                    (0.7, 0.4838569164276123)],
              0.2: [(0.64, 0.9119369983673096), (0.5666666666666667, 0.9882519245147705), (0.84, 0.8055200576782227),
                    (0.6533333333333333, 1.0509471893310547), (0.7066666666666667, 0.7042741775512695)],
              0.3: [(0.8533333333333334, 1.0187041759490967), (0.7733333333333333, 1.2940878868103027),
                    (0.64, 1.5854830741882324),
                    (0.8, 1.0859198570251465), (0.78, 1.0480639934539795)],
              0.4: [(0.7133333333333334, 1.6669230461120605), (0.74, 1.7863783836364746),
                    (0.7333333333333333, 1.75593900680542),
                    (0.7733333333333333, 1.1761667728424072), (0.8466666666666667, 1.294414758682251)],
              0.5: [(0.82, 2.0363831520080566), (0.7933333333333333, 1.8588762283325195),
                    (0.7666666666666667, 1.5538628101348877),
                    (0.7733333333333333, 1.7422380447387695), (0.76, 2.645987033843994)],
              0.6: [(0.8666666666666667, 1.8566420078277588), (0.86, 1.8984110355377197),
                    (0.7333333333333333, 3.024580955505371),
                    (0.8866666666666667, 1.7799358367919922), (0.8733333333333333, 2.030834913253784)],
              0.7: [(0.86, 2.7713167667388916), (0.8733333333333333, 2.2981698513031006),
                    (0.8666666666666667, 2.878854990005493),
                    (0.8333333333333334, 2.4712560176849365), (0.8066666666666666, 3.3563950061798096)],
              0.8: [(0.8, 2.6256840229034424), (0.8, 4.1420512199401855), (0.86, 2.4994890689849854),
                    (0.8666666666666667, 3.177964687347412), (0.88, 3.2199318408966064)],
              0.9: [(0.88, 2.671525001525879), (0.88, 2.852217197418213), (0.8333333333333334, 2.871859073638916),
                    (0.88, 2.7824878692626953), (0.9066666666666666, 3.4080519676208496)],
              1.0: [(0.88, 3.071500778198242), (0.88, 3.0865321159362793), (0.88, 3.107666015625),
                    (0.88, 3.118252992630005),
                    (0.88, 3.0601022243499756)]}


def plot_graph(data, train_size, name):
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