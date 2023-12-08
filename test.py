import openai
import config
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



openai.api_key = config.api_key

encoding = {
    "normal_weight" : 0,
    "obesity" : 1,
    "overweight" : 2,
    "underweight" : 3
}

predicted_output = []
with open("obesity_test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        messages = data["messages"]
        response = openai.chat.completions.create(
            model="ft:gpt-3.5-turbo-0613:personal::8SBEArVW",
            messages=messages,
        )
        predicted_output.append(response)


AI_fine_tuned_pred = []
for response in predicted_output :
    if "normal" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(0)
    elif "overweight" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(2)
    elif "obes" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(1)
    elif "underweight" in ((response.choices[0]).message.content).lower():
        AI_fine_tuned_pred.append(3)
    else:
        AI_fine_tuned_pred.append((response.choices[0]).message.content)

test_df = pd.read_csv("./test_data.csv")

test_df["category_encoding"] = test_df["Category_type"].map(encoding)

print(test_df["category_encoding"].values, len(test_df["category_encoding"].values))
print("-"*150)

print("length of prediction encoded and number of responses back : ", len(AI_fine_tuned_pred), len(predicted_output))
print("-" * 150)

print("Accuracy of the model: ", accuracy_score(test_df["category_encoding"].values, AI_fine_tuned_pred))
print("-" * 150)

print("Confusion Matrix: ", confusion_matrix(test_df["category_encoding"].values, AI_fine_tuned_pred))
print("-" * 150)

print("Classification Report: ", classification_report(test_df["category_encoding"].values, AI_fine_tuned_pred))
print("-" * 150)



print(AI_fine_tuned_pred)