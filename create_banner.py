import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Create a high-quality banner image
fig = plt.figure(figsize=(16, 9), dpi=150)
ax = fig.add_subplot(111)

# Create gradient background (NYC at night theme)
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack([gradient] * 256)

# NYC-inspired colors: dark blue to light blue
colors_custom = plt.cm.Blues(gradient)
ax.imshow(colors_custom, extent=[0, 10, 0, 10], aspect='auto', alpha=0.7)

# Add subtle grid pattern (like city lights)
for i in np.arange(0, 10, 0.5):
    ax.axvline(x=i, color='yellow', alpha=0.1, linewidth=0.5)
    ax.axhline(y=i, color='yellow', alpha=0.1, linewidth=0.5)

# Add some "buildings" - rectangles representing NYC skyline
building_heights = [2, 3, 1.5, 4, 2.5, 3.5, 2, 4.5, 2.5, 3]
for i, height in enumerate(building_heights):
    rect = FancyBboxPatch((i-0.4, 0), 0.8, height, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='gold', facecolor='darkblue', 
                          linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    
    # Add windows
    for j in range(int(height)):
        for k in range(2):
            circle = plt.Circle((i-0.2+k*0.4, j+0.3), 0.08, 
                              color='yellow', alpha=0.9)
            ax.add_patch(circle)

# Add title and subtitle
ax.text(5, 7.5, 'NYC TAXI RIDE DURATION PREDICTION', 
        fontsize=48, fontweight='bold', ha='center', color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='darkblue', alpha=0.7, edgecolor='gold', linewidth=3))

ax.text(5, 5.5, 'A Comprehensive Data Science Analysis of 1.3M Rides', 
        fontsize=28, ha='center', color='lightblue', fontweight='bold')

ax.text(5, 4.5, 'XGBoost • Feature Engineering • Machine Learning • Model Optimization', 
        fontsize=18, ha='center', color='gold', style='italic')

# Add stats
stats_text = 'R² = 0.8146  |  RMSE = 22.8 min  |  MAE = 0.9 min'
ax.text(5, 1, stats_text, fontsize=20, ha='center', color='white', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/rajarshi/projects/new-york-taxirides/assets/banner.jpg', 
            dpi=150, bbox_inches='tight', facecolor='darkblue')
print("✓ Banner created successfully!")
plt.close()
