import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    """
    Handles data loading, cleaning, feature engineering,
    analysis, and visualization.
    """

    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.df = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load dataset with validation."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        if self.df.empty:
            raise ValueError("Loaded dataset is empty")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Handle missing values and enforce correct data types."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Convert types safely
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors='coerce')

        # Handle missing values
        self.df['price'].fillna(self.df['price'].mean(), inplace=True)
        self.df['quantity'].fillna(self.df['quantity'].median(), inplace=True)

        # Drop rows with invalid dates
        self.df.dropna(subset=['date'], inplace=True)

        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        """Create derived features."""
        if self.df is None:
            raise RuntimeError("Data not available for feature engineering.")

        self.df['total_sales'] = self.df['price'] * self.df['quantity']
        return self.df

    def analyze(self) -> dict:
        """Generate summary statistics."""
        if self.df is None:
            raise RuntimeError("Data not ready for analysis.")

        summary = {
            "describe": self.df.describe(),
            "sales_by_product": self.df.groupby('product')['total_sales'].sum()
        }

        return summary

    def visualize(self) -> str:
        """Create and save visualization."""
        if self.df is None:
            raise RuntimeError("Data not ready for visualization.")

        plt.figure(figsize=(8, 5))
        sns.barplot(x='product', y='total_sales', data=self.df)

        plt.title("Total Sales by Product")
        plt.xlabel("Product")
        plt.ylabel("Total Sales")

        output_path = self.output_dir / "sales_by_product.png"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return str(output_path)


def main():
    base_dir = Path(__file__).resolve().parent.parent

    data_path = base_dir / "data" / "sales_data.csv"
    output_dir = base_dir / "outputs"

    processor = DataProcessor(data_path, output_dir)

    processor.load_data()
    processor.clean_data()
    processor.feature_engineering()
    summary = processor.analyze()
    plot_path = processor.visualize()

    print("\n=== Analysis Summary ===")
    print(summary["describe"])
    print("\n=== Sales by Product ===")
    print(summary["sales_by_product"])
    print(f"\nPlot saved at: {plot_path}")


if __name__ == "__main__":
    main()