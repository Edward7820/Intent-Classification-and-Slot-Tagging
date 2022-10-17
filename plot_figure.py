import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_figure(loss_data, accuracy_data):
    loss_df = pd.DataFrame(data=loss_data,dtype=float)
    accuracy_df = pd.DataFrame(data=accuracy_data,dtype=float)
    loss_df.plot()
    accuracy_df.plot()
    plt.show()