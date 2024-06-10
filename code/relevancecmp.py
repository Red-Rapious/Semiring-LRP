import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def compare_relevance(reference, R0, R1, shape=None, label0=None, label1=None):
    difference = R0-R1
    normalized_difference = R0/(R0.mean()) - R1/(R1.mean())
    
    aard = np.absolute(difference).mean()
    naard = 2*np.absolute(difference).mean()/(np.absolute(R0).mean()+np.absolute(R1).mean())

    print(f"Average absolute relevance difference: {aard:.3f}")
    print(f"Normalized average absolute relevance difference: {naard*10:.2f}%")

    reference_rs = reference
    R0_rs = R0
    R1_rs = R1
    if shape is not None:
        reference_rs = reference.reshape(shape)
        R0_rs = R0.reshape(shape)
        R1_rs = R1.reshape(shape)
        difference = difference.reshape(shape)
        normalized_difference = normalized_difference.reshape(shape)

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    
    _, axs = plt.subplots(2,3, figsize=(15,10))

    axs[1][2].set_visible(False)
    axs[1][0].set_position([0.24,0.125,0.228,0.343])
    axs[1][1].set_position([0.55,0.125,0.228,0.343])

    axs[0][0].imshow(reference_rs, cmap="gray")
    axs[0][0].set_title("Reference image")

    b = 10*(np.abs(R0_rs)**3.0).mean()**(1.0/3)
    axs[0][1].imshow(R0_rs, cmap=my_cmap, vmin=-b, vmax=b,interpolation='nearest')
    title = "Relevance 0"
    if label0 is not None:
        title += "\n" + label0
    axs[0][1].set_title(title)

    #b = 10*(np.abs(R1_rs)**3.0).mean()**(1.0/3)
    axs[0][2].imshow(R1_rs, cmap=my_cmap, vmin=-b, vmax=b,interpolation='nearest')
    title = "Relevance 1"
    if label1 is not None:
        title += "\n" + label1
    axs[0][2].set_title(title)

    #b = 10*(np.abs(difference)**3.0).mean()**(1.0/3)
    axs[1][0].imshow(difference,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    axs[1][0].set_title("Relevance difference")

    b = 10*(np.abs(normalized_difference)**3.0).mean()**(1.0/3)
    axs[1][1].imshow(normalized_difference,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    axs[1][1].set_title("Normalized relevance difference")
    
    plt.show()