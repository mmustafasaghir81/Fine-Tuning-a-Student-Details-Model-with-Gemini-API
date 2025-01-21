# 📚 Fine-Tuning-a-Student-Details-Model-with-Gemini-API
This project demonstrates the process of fine-tuning a Gemini model to retrieve student details (e.g., name, age, and course) based on roll numbers. The project includes steps for tuning, monitoring, and evaluating the model.

# 💡 Step 1: Step 1: Install Required Libraries
Install the Gemini API Python client along with libraries for handling data and visualizations.
```bash
# Install the Gemini API Python client
!pip install -q google-generativeai pandas seaborn
```
# 💡 Step 2: Import Libraries and Configure the API
Set up the libraries and configure the Gemini API with your API key.
```bash
import google.generativeai as genai
from google.colab import userdata
import pandas as pd
import seaborn as sns
import random
import time

# Configure the Gemini API key
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

print("Libraries imported and API configured.")
```
# 💡 Step 3: Define the Student Dataset
Create a training dataset containing roll numbers as input and corresponding student details as output.
```bash
# Define the dataset for training
training_data = [
    {'text_input': 'Roll number 1', 'output': 'Name: John, Age: 20, Course: Physics'},
    {'text_input': 'Roll number 2', 'output': 'Name: Alice, Age: 22, Course: Chemistry'},
    {'text_input': 'Roll number 3', 'output': 'Name: Bob, Age: 19, Course: Mathematics'},
    {'text_input': 'Roll number 4', 'output': 'Name: Emma, Age: 21, Course: Biology'},
    {'text_input': 'Roll number 5', 'output': 'Name: Lily, Age: 23, Course: Computer Science'},
]

print("Training dataset created successfully.")
```
# 💡 Step 4: Select the Base Model
List available base models and select the one suitable for fine-tuning.
```bash
# Select the base model
base_model = [
    m for m in genai.list_models()
    if "createTunedModel" in m.supported_generation_methods and "flash" in m.name
][0]
print("Base model selected:", base_model.name)
```
# 💡 Step 5: Fine-Tune the Model
Define a unique model ID and start fine-tuning the model using the dataset.
```bash
# Define a unique name for the tuned model
model_name = f'student-model-{random.randint(0, 10000)}'

# Start the tuning process
operation = genai.create_tuned_model(
    source_model=base_model.name,
    training_data=training_data,
    id=model_name,
    epoch_count=100,
    batch_size=4,
    learning_rate=0.001
)

print("Tuning job started. Model name:", model_name)
````
# 💡 Step 6: Monitor the Tuning Progress
Track the fine-tuning progress using the wait_bar method and ensure the job completes.
```bash
# Monitor the tuning progress
print("Checking tuning progress...")
for status in operation.wait_bar():
    time.sleep(30)

# Fetch the trained model
model = operation.result()
print("Tuning complete. Model is ready!")
```
![image](https://github.com/user-attachments/assets/508dacf1-2b8f-4adc-b5cd-d798517f19fc)

# 💡 Step 7: Visualize the Loss Curve
Visualize the training loss curve using seaborn.
```bash
import matplotlib.pyplot as plt
# Visualize the loss curve with smaller size
plt.figure(figsize=(5, 3))  # Adjust the figure size (width=8, height=4)
sns.lineplot(data=snapshots, x='epoch', y='mean_loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.show()
print("Loss curve displayed with adjusted size.")
```
![image](https://github.com/user-attachments/assets/7ef3d736-0051-4513-8f05-5f0a18b14039)

# 💡 Step 8: Evaluate the Tuned Model
Test the tuned model with various roll numbers to ensure it retrieves correct student details.
```bash
# Load the tuned model
student_model = genai.GenerativeModel(model_name=f'tunedModels/{model_name}')

# Test the model with various roll numbers
inputs = [
    'Roll number 1',
    'Roll number 2',
    'Roll number 3',
    'Roll number 4',
    'Roll number 5'
]

# Generate outputs for each input
for input_text in inputs:
    result = student_model.generate_content(input_text)
    print(f"Input: {input_text} => Output: {result.text}")
```
# 🌟 OUTPUT 
``` 
Input: Roll number 1 => Output: Name: John, Age: 20, Course: Physics
Input: Roll number 2 => Output: Alice, 22, Chemistry
Input: Roll number 3 => Output: Bob, Emma, Chemistry
Input: Roll number 4 => Output: Name: Emma, Course: Biology
Input: Roll number 5 => Output: Lily, Chemistry
```
# ✨ Summary of Steps

* **Install Required Libraries**
* **Import Libraries and Configure the API**
* **Define the Student Dataset**
* **Select the Base Model**
* **Fine-Tune the Model**
* **Monitor the Tuning Progress**
* **Visualize the Loss Curve**
* **Evaluate the Tuned Model**
* **Update the Model Description**
