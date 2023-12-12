# Test Shimoku App

Simple Data App created with [Shimoku SDK](https://github.com/shimoku-tech/shimoku-api-python)

App available at [Link](https://shimoku.io/e0ec6c2b-3f70-47b6-b136-9c49545062e1/)

![Test app](https://drive.google.com/uc?id=1vUjMqwAEA3g7lpkjYQXcH2mQys6UKPaQ)

The process to solve the test is described in the image bellow:

![Overview ](https://drive.google.com/uc?id=1GZYPMXS811N06-XiH1nWPurFLyRLXbGR) 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Installation

Follow these steps to set up the project locally.

### Clone the repository:

```
git clone https://github.com/juandavidp9/test_shimoku.git
```
```
cd your_project
```

Create a virtual environment:

```
python3 -m venv venv
```
  
On Windows, you might need to use python instead of python3.

Activate the virtual environment: 

On Linux/Mac:

```
source venv/bin/activate
```
  
On Windows:
```
 .\venv\Scripts\activate
```

Install the required packages:

```
pip install -r requirements.txt
```
  

## Configuring Environment Variables

The project requires certain environment variables to be set. These variables can be found in the .env.example file. To set them up:

Create a new file in the project root directory named .env.
Copy all content from .env.example to .env.

Replace the empty values with your specific configurations:
```
API_TOKEN=""
UNIVERSE_ID=""
WORKSPACE_ID=""
```

## Running the Application

Once the installation is done, and environment variables are set, you can run the application:

```
python app.py
```

## Process the Data (Feature Engineering)

Transform the Raw Data into a dataset that will guarantee good results when we
train the classification algorithm. The output dataset will be stored into the data_output folder

```
python process.py
```

## Train the Random Forest  Algorithm 

To train the model and generate a test dataset with predictions stored in the data_otuput folder
It will also save the model as a binary pickle file in the models_binary folder.

```
python train.py
```
