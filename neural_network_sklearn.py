"""
Customer Purchase Prediction Neural Network - Fresh Start
Portfolio Project with Guaranteed Visualizations
"""

# Fix matplotlib backend for Mac
import matplotlib
matplotlib.use('Agg')  # This ensures plots are saved as files
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def create_directories():
    """Create project directories"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("✅ Project directories created: data/, models/, results/")

def generate_customer_data(n_samples=1000):
    """Generate synthetic customer data"""
    print(f"\n🔄 Generating {n_samples} customer records...")
    
    np.random.seed(42)
    
    # Generate realistic customer behavior data
    visit_duration = np.random.exponential(scale=5, size=n_samples)
    pages_visited = np.random.poisson(lam=8, size=n_samples)
    
    # Ensure minimum values
    visit_duration = np.maximum(visit_duration, 0.5)
    pages_visited = np.maximum(pages_visited, 1)
    
    # Create purchase probability (more realistic business logic)
    normalized_duration = visit_duration / 20  # Normalize
    normalized_pages = pages_visited / 20      # Normalize
    
    # Combine features with interaction term
    purchase_prob = 0.1 + 0.3 * normalized_duration + 0.4 * normalized_pages + 0.2 * (normalized_duration * normalized_pages)
    purchase_prob = np.clip(purchase_prob, 0, 1)
    
    # Generate purchases
    purchases = np.random.binomial(1, purchase_prob)
    
    # Create dataset
    data = pd.DataFrame({
        'VisitDuration': visit_duration,
        'PagesVisited': pages_visited,
        'Purchase': purchases
    })
    
    purchase_rate = purchases.mean()
    print(f"✅ Data generated! Purchase rate: {purchase_rate:.1%}")
    
    return data

def create_data_visualizations(data):
    """Create and save data exploration visualizations"""
    print("\n📊 Creating data visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Customer Purchase Prediction - Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Visit Duration Distribution
    axes[0, 0].hist(data['VisitDuration'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Visit Duration Distribution')
    axes[0, 0].set_xlabel('Duration (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Pages Visited Distribution  
    axes[0, 1].hist(data['PagesVisited'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Pages Visited Distribution')
    axes[0, 1].set_xlabel('Number of Pages')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Purchase Distribution
    purchase_counts = data['Purchase'].value_counts()
    colors = ['lightcoral', 'lightblue']
    axes[0, 2].pie(purchase_counts.values, labels=['No Purchase', 'Purchase'], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Purchase Distribution')
    
    # 4. Scatter Plot - Purchase Behavior
    no_purchase = data[data['Purchase'] == 0]
    purchase = data[data['Purchase'] == 1]
    
    axes[1, 0].scatter(no_purchase['VisitDuration'], no_purchase['PagesVisited'], 
                      alpha=0.6, c='red', label='No Purchase', s=20)
    axes[1, 0].scatter(purchase['VisitDuration'], purchase['PagesVisited'], 
                      alpha=0.6, c='green', label='Purchase', s=20)
    axes[1, 0].set_xlabel('Visit Duration')
    axes[1, 0].set_ylabel('Pages Visited')
    axes[1, 0].set_title('Purchase Behavior by Features')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Box Plot Comparison
    data_long = pd.melt(data, id_vars=['Purchase'], 
                       value_vars=['VisitDuration', 'PagesVisited'],
                       var_name='Feature', value_name='Value')
    
    purchase_labels = {0: 'No Purchase', 1: 'Purchase'}
    data_long['Purchase_Label'] = data_long['Purchase'].map(purchase_labels)
    
    sns.boxplot(data=data_long, x='Feature', y='Value', hue='Purchase_Label', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Comparison by Purchase')
    axes[1, 1].legend(title='Outcome')
    
    # 6. Correlation Heatmap
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
               square=True, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('results/data_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Data visualization saved: results/data_analysis.png")
    plt.close()

def train_neural_network(data):
    """Train the neural network model"""
    print("\n🧠 Training Neural Network...")
    
    # Prepare data
    X = data[['VisitDuration', 'PagesVisited']].values
    y = data['Purchase'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    
    print("Model Architecture: Input(2) → Hidden(32) → Hidden(16) → Hidden(8) → Output(1)")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Training complete!")
    print(f"   Training iterations: {model.n_iter_}")
    print(f"   Test Accuracy: {accuracy:.3f}")
    print(f"   AUC Score: {auc_score:.3f}")
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba

def create_model_visualizations(y_test, y_pred, y_pred_proba):
    """Create and save model performance visualizations"""
    print("\n📈 Creating model performance visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Neural Network Model Performance', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction Distribution
    axes[1, 0].hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance Metrics Summary
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"""Performance Metrics:

Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1-Score: {f1:.3f}
AUC: {auc:.3f}

Confusion Matrix:
True Positives: {tp}
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
"""
    
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Performance Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/model_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Model visualization saved: results/model_performance.png")
    plt.close()

def demo_predictions(model, scaler):
    """Demonstrate predictions on sample customers"""
    print("\n🎯 Demo Predictions:")
    print("-" * 50)
    
    # Sample customers
    customers = np.array([
        [2.0, 3],    # Low engagement
        [8.0, 12],   # Medium engagement  
        [15.0, 20],  # High engagement
        [1.0, 1],    # Very low engagement
        [25.0, 30]   # Very high engagement
    ])
    
    labels = ["Low Engagement", "Medium Engagement", "High Engagement", 
              "Very Low Engagement", "Very High Engagement"]
    
    # Scale and predict
    customers_scaled = scaler.transform(customers)
    probabilities = model.predict_proba(customers_scaled)[:, 1]
    
    for i, (customer, prob, label) in enumerate(zip(customers, probabilities, labels)):
        print(f"Customer {i+1} ({label}):")
        print(f"  Duration: {customer[0]:.1f} min, Pages: {customer[1]:.0f}")
        print(f"  Purchase Probability: {prob:.3f} ({prob*100:.1f}%)")
        print()

def main():
    """Main function - runs the complete neural network pipeline"""
    print("🚀 CUSTOMER PURCHASE PREDICTION NEURAL NETWORK")
    print("=" * 60)
    
    # Setup
    create_directories()
    
    # Generate data
    data = generate_customer_data(1000)
    
    # Save data
    data.to_csv('data/customer_data.csv', index=False)
    print("✅ Data saved to: data/customer_data.csv")
    
    # Create data visualizations
    create_data_visualizations(data)
    
    # Train model
    model, scaler, X_test, y_test, y_pred, y_pred_proba = train_neural_network(data)
    
    # Create model visualizations
    create_model_visualizations(y_test, y_pred, y_pred_proba)
    
    # Demo predictions
    demo_predictions(model, scaler)
    
    # Final summary
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("🎉 NEURAL NETWORK PROJECT COMPLETE!")
    print("=" * 60)
    print(f"✅ Final Accuracy: {accuracy:.3f}")
    print(f"✅ Final AUC Score: {auc:.3f}")
    print(f"✅ Visualizations saved in results/ folder")
    print(f"✅ Data saved in data/ folder")

if __name__ == "__main__":
    main()