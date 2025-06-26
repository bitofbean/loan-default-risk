import pandas as pd
import matplotlib.pyplot as plt
import os

# Set up paths
MONITOR_DIR = "/opt/airflow/monitoring"
LOG_FILE = os.path.join(MONITOR_DIR, "accuracy_log.csv")
OUTPUT_DIR = os.path.join(MONITOR_DIR, "visualizations")

def create_simple_accuracy_plot():
    """Create a simple accuracy trend plot"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load accuracy log
    if not os.path.exists(LOG_FILE):
        print(f"ERROR: Accuracy log not found: {LOG_FILE}")
        return
    
    df = pd.read_csv(LOG_FILE)
    
    # Convert date to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['accuracy'], marker='o', linewidth=2, markersize=8, color='blue')
    
    # Formatting
    plt.title('Model Accuracy Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on points
    for date, acc in zip(df['date'], df['accuracy']):
        plt.annotate(f'{acc:.1%}', (date, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Set y-axis to show meaningful range
    plt.ylim(0.68, 0.74)
    
    # Format x-axis to show dates as YYYY-MM-DD (first day of month format)
    plt.xticks(df['date'], [date.strftime('%Y-%m-%d') for date in df['date']], rotation=45)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(OUTPUT_DIR, "accuracy_trend.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to: {output_file}")
    
    # Show summary
    print("\nMONITORING SUMMARY:")
    print(f"Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Average Accuracy: {df['accuracy'].mean():.1%}")
    print(f"Best Performance: {df['accuracy'].max():.1%}")
    print(f"Latest Performance: {df['accuracy'].iloc[-1]:.1%}")
    
    if len(df) > 1:
        trend = "improving" if df['accuracy'].iloc[-1] > df['accuracy'].iloc[0] else "stable/declining"
        improvement = df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]
        print(f"Trend: {trend} (+{improvement:.1%})" if improvement > 0 else f"Trend: {trend} ({improvement:.1%})")

if __name__ == "__main__":
    print("Creating simple accuracy monitoring plot...")
    create_simple_accuracy_plot()