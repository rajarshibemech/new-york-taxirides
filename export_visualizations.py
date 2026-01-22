import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data from the notebook environment (assuming notebook kernel is active)
import sys
sys.path.insert(0, '/Users/rajarshi/projects/new-york-taxirides')

# High-quality plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("Exporting visualizations from notebook...")

# Create visualizations that will be called from the notebook kernel
export_code = """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set high quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 1. Feature Importance Plot
fig, ax = plt.subplots(figsize=(12, 6))
importance_df_sorted = importance_df.sort_values('Importance', ascending=True)
colors = sns.color_palette("viridis", len(importance_df_sorted))
ax.barh(importance_df_sorted['Feature'], importance_df_sorted['Importance'], color=colors)
ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('XGBoost Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/rajarshi/projects/new-york-taxirides/assets/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved feature_importance.png")

# 2. Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test_xgb, y_pred_test, alpha=0.5, s=20)
ax.plot([y_test_xgb.min(), y_test_xgb.max()], [y_test_xgb.min(), y_test_xgb.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Duration (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Duration (seconds)', fontsize=12, fontweight='bold')
ax.set_title(f'XGBoost: Actual vs Predicted (R² = {test_r2:.4f})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/rajarshi/projects/new-york-taxirides/assets/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved actual_vs_predicted.png")

# 3. Residuals Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals histogram
axes[0].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[0].set_xlabel('Residuals (seconds)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals vs Predicted
axes[1].scatter(y_pred_test, residuals_test, alpha=0.5, s=20)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Duration (seconds)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (seconds)', fontsize=12, fontweight='bold')
axes[1].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/rajarshi/projects/new-york-taxirides/assets/residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved residuals_analysis.png")

# 4. Model Performance Metrics
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['R² Score', 'RMSE', 'MAE']
train_vals = [train_r2, train_rmse, train_mae]
test_vals = [test_r2, test_rmse, test_mae]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, train_vals, width, label='Training', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='coral', edgecolor='black')

ax.set_ylabel('Score / Error', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/rajarshi/projects/new-york-taxirides/assets/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved model_performance.png")

# 5. Cross-Validation Scores
fig, ax = plt.subplots(figsize=(10, 6))
cv_folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
ax.bar(cv_folds, cv_scores, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax.axhline(y=np.mean(cv_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cv_scores):.4f}')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([np.mean(cv_scores) - 0.05, np.mean(cv_scores) + 0.05])

plt.tight_layout()
plt.savefig('/Users/rajarshi/projects/new-york-taxirides/assets/cross_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved cross_validation.png")

print("\\nAll visualizations exported successfully!")
"""

print(export_code)
