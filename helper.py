

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, epsilonHistory, QVals, lossVals):
    plt.figure(1)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.figure(2)
    plt.title("Epsilon decay graph")
    plt.ylabel('Epsilon')
    plt.xlabel('Number of Games')    
    plt.plot(epsilonHistory)

    plt.figure(3)
    plt.title("Q values graph")
    plt.ylabel('Qvalue')
    plt.xlabel('Number of Batches')   
    plt.plot(QVals)

    plt.figure(4)
    plt.title("Loss values graph")
    plt.ylabel('loss')
    plt.xlabel('Number of Batches')   
    plt.plot(lossVals)

    #plt.plot(regression_line)

    plt.show(block=False)
    plt.pause(.1)
