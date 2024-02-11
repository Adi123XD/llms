# Blog Generation 
## Step 1: Run the following commands 
```
conda create -p venv python==3.10.11 (or whatever your version is)
conda activate path_to_your_venv
pip install -r requirements.txt
```
after running the commands load the llama 2 model inside the model directory
go to 
[TheBloke/Llama-2-7B-GGML · Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-GGML)
and load the mode using the command 
```
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-GGML
```

## Step 2: Run app.py
after installing all the dependencies we can open the streamlit website using the command below in the terminal 
```
streamlit run app.py 
```

and the website will open in new terminal 
easy peasy