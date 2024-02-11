# Gemini 
## Step 1:  Create google pro api key 
Firstly we need to create an api key to use the google gemini model. To create the api key 
go to     [Get API key ](https://makersuite.google.com/app/apikey) 
and create the api key 

## Step 2: Run the following commands 
```
conda create -p venv python==3.10.11 (or whatever your version is)
conda activate path_to_your_venv
pip install -r requirements.txt
```

## Step 3: Run chat.py
after installing all the dependencies we can open the streamlit website using the command below in the terminal 
```
streamlit run chat.py 
```

and the website will open in new terminal 
easy peasy
references taken from : [Here](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/tutorials/python_quickstart.ipynb?authuser=1#scrollTo=gHYFrFPjSGNq)