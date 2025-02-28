{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "61c60b70-f8eb-4399-ba14-f6ad2f4e62b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.datasets import load_iris\n",
    "import plotly.express as px\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "cac94be6-559c-4942-8559-d6b5168de02c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def kmeans_clustering():\n",
    "    iris = load_iris()\n",
    "    df = pd.DataFrame(iris.data, columns=iris.feature_names)\n",
    "\n",
    "    scaler = StandardScaler()\n",
    "    df_scaled = scaler.fit_transform(df)\n",
    "\n",
    "    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n",
    "    df['Cluster'] = kmeans.fit_predict(df_scaled)\n",
    "\n",
    "    return df, iris.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "6c61409d-5821-4a99-97d7-638ddf4de63b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_2d_scatter(df):\n",
    "    plt.figure(figsize=(8, 6))\n",
    "\n",
    "    sns.scatterplot(x=df['sepal length (cm)'], \n",
    "                    y=df['sepal width (cm)'], \n",
    "                    hue=df['Cluster'], \n",
    "                    palette='viridis')\n",
    "\n",
    "    plt.xlabel(\"Sepal Length (cm)\")\n",
    "    plt.ylabel(\"Sepal Width (cm)\")\n",
    "    plt.title(\"K-Means Clustering (2D View)\")\n",
    "\n",
    "    plt.savefig(\"static/plot_2d.png\")\n",
    "    plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "4df89187-5fe8-46dd-b5bf-6b8b1f80cc0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_3d_scatter(df):\n",
    "    fig = px.scatter_3d(df, \n",
    "                        x='sepal length (cm)', \n",
    "                        y='sepal width (cm)', \n",
    "                        z='petal length (cm)',  # Fixed the 'z' argument\n",
    "                        color=df['Cluster'].astype(str),  # Fixed bracket typo\n",
    "                        title=\"K-Means Clustering (3D View)\")\n",
    "\n",
    "    fig.write_html(\"static/plot_3d.html\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "17a6261e-3609-4022-abb0-0c0a77ef5a7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df,sample = kmeans_clustering()\n",
    "plot_2d_scatter(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46e5fd6f-9f03-41f6-8432-bb361817698b",
   "metadata": {},
   "outputs": [],
   "source": [
    "90"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "8e35afab-791d-46cf-ade9-8217e4f45077",
   "metadata": {},
   "outputs": [],
   "source": [
    "df,sample = kmeans_clustering()\n",
    "plot_3d_scatter(df)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
