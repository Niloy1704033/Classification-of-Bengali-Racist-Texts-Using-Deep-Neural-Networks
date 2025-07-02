# CV Classifier and Sorter Deep Learning Project

import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

# 1. Sample Data (Replace this with more comprehensive JSON input from CVs)
data = [
    {
        "name": "Arif Rahman",
        "education": "BSc in Electrical Engineering",
        "skills": ["C++", "Embedded Systems", "IoT"],
        "experience": "3 years in hardware programming",
        "summary": "Skilled in microcontroller-based systems.",
        "category": "Embedded Systems"
    },
    {
        "name": "Tahmina Akter",
        "education": "MSc in Data Science",
        "skills": ["Python", "Pandas", "Scikit-learn", "NLP"],
        "experience": "2 years as a data analyst at a fintech company",
        "summary": "Passionate about AI and analytics.",
        "category": "Data Science"
    },
    {
        "name": "Mehedi Hasan",
        "education": "BSc in Software Engineering",
        "skills": ["JavaScript", "React", "Node.js"],
        "experience": "1 year in full-stack development",
        "summary": "Love building web apps.",
        "category": "Web Development"
    }
]

# 2. Data Preprocessing

def combine_fields(cv):
    return f"{cv['education']} {' '.join(cv['skills'])} {cv['experience']} {cv['summary']}"

texts = [combine_fields(cv) for cv in data]
categories = [cv['category'] for cv in data]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CVDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Dataset split
X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)
train_dataset = CVDataset(X_train, y_train)
test_dataset = CVDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 3. Model Definition

class CVClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CVClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# 4. Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVClassifier(num_classes=len(label_encoder.classes_)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
preds, true = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1)
        preds.extend(pred.cpu().numpy())
        true.extend(labels.cpu().numpy())

print(classification_report(true, preds, target_names=label_encoder.classes_))

# 6. Inference Example
def predict_category(cv):
    text = combine_fields(cv)
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        output = model(tokens['input_ids'], tokens['attention_mask'])
        pred_class = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred_class])[0]

# Sort CVs by category match
def sort_cvs_by_category(cvs, target_category):
    scored = []
    for cv in cvs:
        predicted = predict_category(cv)
        relevance = int(predicted == target_category)
        scored.append((cv['name'], predicted, relevance))
    return sorted(scored, key=lambda x: x[2], reverse=True)

# Example sorting
sorted_candidates = sort_cvs_by_category(data, "Data Science")
print(json.dumps(sorted_candidates, indent=2))
