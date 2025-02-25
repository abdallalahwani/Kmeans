import matplotlib
# i dont know what that does but it didnt work without it :)
matplotlib.use('Agg')  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def plot_elbow_method():
    data = load_iris()
    X = data.data

    K_range = range(1, 11)
    inertias = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

   
    percent_changes = np.diff(inertias) / inertias[:-1] * -100
    if len(percent_changes) > 2:
        # ignore first and last elements
        valid_range = percent_changes[1:-1]  
        elbow_k = np.argmax(valid_range) + 2  
    else:
        #if cant exclude boundaries 
        elbow_k = np.argmax(percent_changes) + 1  

    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(K_range)
    plt.grid(True)
    
    # Adjust annotation for the elbow point
    plt.annotate('Elbow Point', xy=(elbow_k, inertias[elbow_k - 1]),
                 xytext=(elbow_k + 15, inertias[elbow_k - 1] -100 ),  
                 textcoords="offset points",
                 ha='center',
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.5))  

    plt.savefig("elbow.png")
    plt.close()

if __name__ == "__main__":
    plot_elbow_method()
