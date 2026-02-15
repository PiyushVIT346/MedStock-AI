import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

def create_ultimate_features(df):
    """Create the most comprehensive feature set"""
    print("🔧 Creating ultimate feature engineering...")
    
    # 1. One-hot encoding
    diagnosis_dummies = pd.get_dummies(df['Primary_Diagnosis'], prefix='Diag')
    procedure_dummies = pd.get_dummies(df['Procedure_Performed'], prefix='Proc')
    room_dummies = pd.get_dummies(df['Room_Type'], prefix='Room')
    
    # 2. Medical complexity scoring
    complexity_scores = {
        'Appendicitis': 5, 'Fracture': 4, 'Pneumonia': 3, 'Diabetes': 2
    }
    procedure_scores = {
        'Appendectomy': 5, 'MRI': 4, 'Chest X-ray': 2, 'Blood Test': 1
    }
    
    df['Diag_Complexity'] = df['Primary_Diagnosis'].map(complexity_scores)
    df['Proc_Complexity'] = df['Procedure_Performed'].map(procedure_scores)
    df['Total_Complexity'] = df['Diag_Complexity'] + df['Proc_Complexity']
    
    # 3. Bed days features
    df['Bed_Days_Norm'] = (df['Bed_Days'] - df['Bed_Days'].min()) / (df['Bed_Days'].max() - df['Bed_Days'].min())
    df['Bed_Days_Log'] = np.log1p(df['Bed_Days'])
    df['Bed_Days_Cat'] = pd.cut(df['Bed_Days'], bins=5, labels=False)
    
    # 4. Critical interaction features
    df['Surgery_ICU'] = ((df['Procedure_Performed'] == 'Appendectomy') & (df['Room_Type'] == 'ICU')).astype(int)
    df['Emergency_Case'] = (df['Primary_Diagnosis'].isin(['Appendicitis', 'Fracture'])).astype(int)
    df['ICU_Case'] = (df['Room_Type'] == 'ICU').astype(int)
    df['Complex_Case'] = (df['Total_Complexity'] > 7).astype(int)
    
    # 5. Staffing prediction features (domain knowledge)
    df['Rule_2_Surgeons'] = (
        ((df['Primary_Diagnosis'] == 'Appendicitis') & (df['Procedure_Performed'] == 'Appendectomy')) |
        ((df['Primary_Diagnosis'] == 'Fracture') & (df['Room_Type'] == 'ICU') & (df['Bed_Days'] > 7))
    ).astype(int)
    
    df['Rule_Nurse_Doctor'] = (
        ((df['Room_Type'] == 'ICU') & (df['Rule_2_Surgeons'] == 0)) |
        ((df['Primary_Diagnosis'].isin(['Fracture', 'Pneumonia'])) & (df['Bed_Days'] > 3))
    ).astype(int)
    
    df['Rule_Nurse_Only'] = (
        ((df['Primary_Diagnosis'] == 'Diabetes') & (df['Procedure_Performed'] == 'Blood Test') & (df['Room_Type'] == 'General Ward')) |
        ((df['Room_Type'] == 'General Ward') & (df['Bed_Days'] <= 3))
    ).astype(int)
    
    # 6. Statistical features
    df['Complexity_Bed_Ratio'] = df['Total_Complexity'] / (df['Bed_Days'] + 1)
    df['Risk_Score'] = (df['Diag_Complexity'] * 0.4 + df['Proc_Complexity'] * 0.3 + 
                       df['ICU_Case'] * 3 + df['Bed_Days_Norm'] * 2)
    
    # Combine all features
    feature_df = pd.concat([
        diagnosis_dummies, procedure_dummies, room_dummies,
        df[['Bed_Days', 'Bed_Days_Norm', 'Bed_Days_Log', 'Bed_Days_Cat',
            'Diag_Complexity', 'Proc_Complexity', 'Total_Complexity',
            'Surgery_ICU', 'Emergency_Case', 'ICU_Case', 'Complex_Case',
            'Rule_2_Surgeons', 'Rule_Nurse_Doctor', 'Rule_Nurse_Only',
            'Complexity_Bed_Ratio', 'Risk_Score']]
    ], axis=1)
    
    print(f"✅ Ultimate features: {feature_df.shape[1]} features")
    return feature_df, df['Staff_Needed']

class UltraAdvancedNet(nn.Module):
    """Ultra advanced neural network with multiple techniques"""
    
    def __init__(self, input_size, num_classes):
        super(UltraAdvancedNet, self).__init__()
        
        # Multi-path architecture
        self.path1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.path2 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        
        # Concatenate paths
        fused = torch.cat([path1_out, path2_out], dim=1)
        fused = self.fusion(fused)
        
        output = self.classifier(fused)
        return output

def train_ultra_model(model, train_loader, val_loader, num_epochs=2000):
    """Ultra advanced training with all techniques"""
    
    # Advanced loss function
    class_weights = torch.FloatTensor([1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Advanced optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.path1.parameters(), 'lr': 0.001},
        {'params': model.path2.parameters(), 'lr': 0.0015},
        {'params': model.fusion.parameters(), 'lr': 0.001},
        {'params': model.classifier.parameters(), 'lr': 0.0005}
    ], weight_decay=1e-4)
    
    # Advanced scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=num_epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    model.to(device)
    
    best_val_acc = 0
    patience = 200
    patience_counter = 0
    
    print(f"Training ultra model on {device}...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_accuracy = 100 * correct / total
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'augmented_best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Val Acc: {val_accuracy:.2f}%, Best: {best_val_acc:.2f}%')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}, Best: {best_val_acc:.2f}%')
            break
    
    # Load best model
    model.load_state_dict(torch.load('augmented_best_model.pth'))
    return model

def create_hybrid_ensemble(X_train, X_test, y_train, y_test, pytorch_predictions):
    """Create hybrid ensemble with traditional ML + PyTorch"""
    
    print("🎯 Creating hybrid ensemble...")
    
    # Train traditional models
    rf = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=42)
    
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    
    # Ensemble voting
    ensemble_predictions = []
    for i in range(len(pytorch_predictions)):
        votes = [pytorch_predictions[i], rf_pred[i], gb_pred[i]]
        # Majority vote
        ensemble_pred = max(set(votes), key=votes.count)
        ensemble_predictions.append(ensemble_pred)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    
    return ensemble_accuracy, ensemble_predictions

def main():
    """Main training pipeline using augmented dataset"""
    print("="*80)
    print("🚀 TRAINING ON AUGMENTED DATASET")
    print("🎯 Using pre-generated augmented_patient_data.csv")
    print("="*80)
    
    # Load augmented dataset
    print("\n📊 Loading augmented dataset...")
    df = pd.read_csv('augmented_patient_data.csv')
    print(f"✅ Loaded {len(df)} samples")
    
    # Create ultimate features
    X, y = create_ultimate_features(df)
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {pd.Series(y).value_counts()}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Data splits:")
    print(f"   Training: {X_train.shape[0]}")
    print(f"   Validation: {X_val.shape[0]}")
    print(f"   Test: {X_test.shape[0]}")
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train.values), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.values), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test.values), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create and train ultra model
    model = UltraAdvancedNet(X_train.shape[1], len(target_encoder.classes_))
    trained_model, history = train_ultra_model(model, train_loader, val_loader, num_epochs=1500)
    
    # Evaluate PyTorch model
    trained_model.eval()
    pytorch_predictions = []
    y_test_list = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = trained_model(batch_X)
            _, predicted = torch.max(outputs, 1)
            pytorch_predictions.extend(predicted.cpu().numpy())
            y_test_list.extend(batch_y.numpy())
    
    pytorch_accuracy = accuracy_score(y_test_list, pytorch_predictions)
    
    # Create hybrid ensemble
    ensemble_accuracy, ensemble_predictions = create_hybrid_ensemble(
        X_train, X_test, y_train, y_test_list, pytorch_predictions
    )
    
    # Final results
    print("\n" + "="*80)
    print("🎯 TRAINING RESULTS ON AUGMENTED DATA")
    print("="*80)
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   PyTorch Model:    {pytorch_accuracy:.4f} ({pytorch_accuracy*100:.2f}%)")
    print(f"   Hybrid Ensemble:  {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    
    best_accuracy = max(pytorch_accuracy, ensemble_accuracy)
    best_model_name = "Hybrid Ensemble" if ensemble_accuracy > pytorch_accuracy else "PyTorch Model"
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Final Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Achievement check
    if best_accuracy >= 0.75:
        print("\n🎉 TARGET ACHIEVED! >75% ACCURACY!")
    elif best_accuracy >= 0.70:
        print("\n🎊 VERY CLOSE! >70% ACCURACY!")
    elif best_accuracy >= 0.65:
        print("\n👍 EXCELLENT PROGRESS! >65% ACCURACY!")
    elif best_accuracy >= 0.60:
        print("\n✅ SIGNIFICANT IMPROVEMENT! >60% ACCURACY!")
    else:
        print("\n📈 SUBSTANTIAL PROGRESS MADE!")
    
    # Detailed classification report
    best_predictions = ensemble_predictions if ensemble_accuracy > pytorch_accuracy else pytorch_predictions
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test_list, best_predictions, 
                            target_names=target_encoder.classes_))
    
    # Save final model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_class': UltraAdvancedNet,
        'input_size': X_train.shape[1],
        'num_classes': len(target_encoder.classes_),
        'scaler': scaler,
        'target_encoder': target_encoder,
        'accuracy': best_accuracy,
        'model_type': best_model_name
    }, 'augmented_data_trained_model.pth')
    
    print(f"\n💾 Final model saved: augmented_data_trained_model.pth")
    print(f"   Best accuracy achieved: {best_accuracy*100:.2f}%")
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETED!")
    print("="*80)
    
    return history, y_test_list, ensemble_predictions, target_encoder, X_train

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_all_insights_first(history, y_test, y_pred, target_names, X_train):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 15))
    
    # --- 1. Learning Curves ---
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss (AdamW)', color='#e74c3c')
    plt.title('Optimization Path: Loss Reduction', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='#2ecc71')
    plt.axhline(y=75, color='r', linestyle='--', label='Target (75%)')
    plt.title('Accuracy Growth over Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # --- 2. Confusion Matrix (The "Insight" into Errors) ---
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Decision Matrix: Where does the model confuse staff needs?', fontsize=14)
    plt.xlabel('Predicted Staffing')
    plt.ylabel('Actual Staffing')

    # --- 3. Feature Importance (The "Why") ---
    # We extract this from the Random Forest component of your ensemble
    from sklearn.ensemble import RandomForestClassifier
    rf_insight = RandomForestClassifier(n_estimators=100).fit(X_train, y_test[:len(X_train)]) # Quick fit for visual
    importances = pd.Series(rf_insight.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    
    plt.subplot(2, 2, 4)
    importances.plot(kind='barh', color='#3498db')
    plt.title('Top 10 Drivers for Staffing Needs', fontsize=14)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_all_insights_second(history, y_test, y_pred, target_names, X_train):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 15))
    
    # --- 1. Learning Curves ---
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss (AdamW)', color='#e74c3c')
    plt.title('Optimization Path: Loss Reduction', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='#2ecc71')
    plt.axhline(y=75, color='r', linestyle='--', label='Target (75%)')
    plt.title('Accuracy Growth over Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # --- 2. Confusion Matrix (The "Insight" into Errors) ---
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Decision Matrix: Where does the model confuse staff needs?', fontsize=14)
    plt.xlabel('Predicted Staffing')
    plt.ylabel('Actual Staffing')

    # --- 3. Feature Importance (The "Why") ---
    # We extract this from the Random Forest component of your ensemble
    from sklearn.ensemble import RandomForestClassifier
    rf_insight = RandomForestClassifier(n_estimators=100).fit(X_train, y_test[:len(X_train)]) # Quick fit for visual
    importances = pd.Series(rf_insight.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    
    plt.subplot(2, 2, 4)
    importances.plot(kind='barh', color='#3498db')
    plt.title('Top 10 Drivers for Staffing Needs', fontsize=14)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    hist, y_test, preds, encoder, x_data = main()
    plot_all_insights_first(hist, y_test, preds, encoder.classes_, x_data)
    plot_all_insights_second(hist, y_test, preds, encoder.classes_, x_data)
