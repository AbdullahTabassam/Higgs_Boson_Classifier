Higgs Boson Classification Model

This is a GitHub repository containing a classification model for Higgs boson particle detection. The dataset used for this project is the Higgs Boson Machine Learning Challenge dataset, which is available on kaggle and https://archive.ics.uci.edu/ml/datasets/HIGGS.

The model is implemented in Python and is available in three Jupyter notebooks, namely EDA.ipynb, Training.ipynb, and Evaluation.ipynb. A fourth notebook, Combined.ipynb, contains the combined code of all three notebooks.

The model training is based on Random Forest and XGBoost models using GPU with CUDA libraries cuml, cudf, and cupy.
Getting Started

To get started with this project, you can either download or clone the repository.
Prerequisites

The following packages are required to run the code:

    Python 3.7 or above
    Jupyter Notebook
    NumPy
    Pandas
    Scikit-learn
    Matplotlib
    Seaborn
    cuml
    cudf
    cupy


Once you have installed the required packages and CUDA libraries, you can open the Jupyter notebooks and run the code.

    EDA.ipynb: This notebook contains the code for exploratory data analysis, where the dataset is loaded, cleaned, and visualized.

    Training.ipynb: This notebook contains the code for training the classification model using Random Forest and XGBoost models with GPU acceleration. The dataset is split into training and testing sets, and the models are trained using the training set. The trained models are then saved to a file.

    Evaluation.ipynb: This notebook contains the code for evaluating the performance of the trained models on the testing set. The accuracy, precision, recall, and F1 score are calculated, and a confusion matrix is plotted.

    Combined.ipynb: This notebook contains the combined code of all three notebooks.

Acknowledgments

    The Higgs Boson Machine Learning Challenge dataset was obtained from the ATLAS experiment challenge present at kaggle as well as:

    https://archive.ics.uci.edu/ml/datasets/HIGGS

    The project was implemented as a part of a machine learning course.