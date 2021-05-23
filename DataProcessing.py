from matplotlib import pyplot as plt
import numpy as np

def readFile(filename):
    f = open(filename, "r")
    try:
        output = f.readlines()
    finally:
        f.close()

    return output

def plot(x, y):
    fig = plt.figure()
    plt.plot(x, y)
    plt.ylabel('Average rewards')
    plt.xlabel('Episodes')
    plt.show()
    fig.savefig("AVR_LR_70.jpg")

if __name__ == "__main__":
    total_rewards = readFile("./GraphsData/DDQN_Reward_LR_0_7.txt")
    average = []
    nmr = []
    temp = []
    for index, (r) in enumerate(total_rewards):
        temp.append(r)
        if index % 5 == 0 and index != 0:
            avr = np.mean(np.array(temp).astype(np.float))
            average.append(avr)
            nmr.append(index)
            temp = []

    plot(nmr, average)