#### Overview
This code is designed to cluster clients based on their bank transaction data using an unsupervised learning approach, specifically the K-Means clustering algorithm. The goal is to group clients into clusters that exhibit similar transaction behaviors, which can then be analyzed for patterns such as potential age groups or spending habits.


### Running the Code
## Running using ducker
Pull and build the container image from GitHub on your command-line interpreter

`docker pull ghcr.io/sadeqebrahim/myapp:latest`

`docker run -d -p 8080:8080 ghcr.io/sadeqebrahim/myapp:latest`

and then go to this localhost: http://localhost:8080/ 


## Running using the notebook
To run this code, follow these steps:

1. **Set Up the Environment**: Ensure that you have a Python environment set up with the necessary libraries installed, including `pandas`, `scikit-learn`, `seaborn`, and `matplotlib`.

2. **Data Preparation**: Place the required CSV files (transactions data, client IDs, etc.) in the appropriate directories specified in the code.

3. **Execute the Script**: Run the Python script. This can be done in an integrated development environment (IDE) like Jupyter Notebook, VSCode, or directly in a Python interpreter.

4. **Visualize and Analyze**: The script will generate visual plots that help in understanding the clusters. It will also save the cluster assignments to a CSV file for further use.

By following these steps, you will be able to cluster clients based on their transaction data, providing valuable insights into their spending behaviors and potentially identifying distinct groups such as age categories or spending profiles.
