import pickle 
import torch 
from model.models import HybridClassifier 
import yaml # Load Config 

def load_config(path): 
    with open(path, "r") as f: 
        return yaml.safe_load(f) 
    
config = load_config("configs/config.yaml") 

# Load Model 
model = HybridClassifier(config=config, use_quantum=False) 
state = torch.load("models/model_backbone_tcga_quantum_78_epochs_20251209_135726.pt", map_location="cpu") 

new_state = {"backbone."+k: v for k, v in state.items()} 
model.load_state_dict(new_state, strict=False) 
model.eval() 

# Testing 
with open("src/data/tcga/testdata.pkl", "rb") as f: 
    data = pickle.load(f) 
    
X_test = data["X"] 
Y_test = data["Y"] 

# Function for predicting 
def predict_embedding(emb): 
    emb_t = torch.tensor(emb).float().unsqueeze(0) 
    with torch.no_grad(): 
        out = model(emb_t)
        probs = torch.softmax(out, dim=1)[0] 
        pred_class = torch.argmax(probs).item() 
    return pred_class, probs[pred_class].item() 

# Import the diseases and see if the labels are the same 
with open("src/data/tcga/traindata.pkl", "rb") as f: 
    train_data = pickle.load(f) 

unique_classes = sorted(list(set(train_data["Y"]))) 
    
N_correct = 0 # Number of correct predicted labels
    
for i in range(len(X_test)): 
    pred, conf = predict_embedding(X_test[i]) 
    class_to_idx = {c: i for i, c in enumerate(unique_classes)} 
    pred_label = unique_classes[pred]
    if pred_label == Y_test[i]: 
        N_correct += 1 
    
prec_correct = (N_correct/len(Y_test))*100 
print(f'Precentage of correct predictions: {prec_correct:.1f}%')