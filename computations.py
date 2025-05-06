import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.markers import MarkerStyle

class DataProcessor:
    def __init__(self, sales_file_path, current_file_path):
        self.sales_file_path = sales_file_path
        self.current_file_path = current_file_path
        self.merged_data = None

    def load_and_merge_data(self):
        # Read CSV files
        sales_data = pd.read_csv(self.sales_file_path)
        current_data = pd.read_csv(self.current_file_path)

        # Merge data if necessary (assuming common columns exist)
        self.merged_data = pd.concat([sales_data, current_data], ignore_index=True)
        self.merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['SYEAR', 'SMONTH'])
        self.merged_data = self.merged_data[self.merged_data['QUALIFIED'] == 'Q']
        self.merged_data['Unadjusted Ratio'] = self.merged_data['TASP'] / self.merged_data['CURRTOT']


        # Calculate base year and base month
        base_year = self.merged_data['SYEAR'].min()
        base_month = self.merged_data.loc[self.merged_data['SYEAR'] == base_year, 'SMONTH'].min()

        # Compute 'MONTHS' column
        self.merged_data['MONTHS'] = ((self.merged_data['SYEAR'] - base_year) * 12) + (self.merged_data['SMONTH'] - base_month)

        # Ensure chronological increments (set base month to 0)
        self.merged_data['MONTHS'] = self.merged_data['MONTHS'] - self.merged_data['MONTHS'].min()

        # Sort the data by the 'MONTHS' column in ascending order
        self.merged_data = self.merged_data.sort_values(by='MONTHS', ascending=True)
        

        return self.merged_data

    def generate_scatter_plot(self, usetype, merged_data):
        # Replace infinities and drop NaN entries for relevant columns
        merged_data = self.merged_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['SYEAR', 'SMONTH', 'Unadjusted Ratio'])

        # Apply filter for Qualified == 'Q'
        merged_data = merged_data[merged_data['QUALIFIED'] == 'Q']

        # Define colors for each USETYPE
        colors = {
            'VACANT LAND': 'black',
            'RESIDENTIAL': 'green',
            'COMMERCIAL': 'blue'
        }

        # Calculate the base year and base month
        base_year = merged_data['SYEAR'].min()
        base_month = merged_data.loc[merged_data['SYEAR'] == base_year, 'SMONTH'].min()

        # Compute 'MONTHS' column
        merged_data['MONTHS'] = ((merged_data['SYEAR'] - base_year) * 12) + (merged_data['SMONTH'] - base_month)

        # Ensure chronological increments (set base month to 0)
        merged_data['MONTHS'] = merged_data['MONTHS'] - merged_data['MONTHS'].min()

        # Sort the merged data by 'MONTHS' to maintain chronological order
        merged_data = merged_data.sort_values(by='MONTHS', ascending=True)

        # Debug: Ensure MONTHS column is populated correctly
        if merged_data['MONTHS'].isna().any():
            raise ValueError("The 'MONTHS' column contains NaN values after calculation.")

        # Set up figure for combined scatter plot
        plt.figure(figsize=(16, 8))

        # Iterate through each unique USETYPE and plot on the same graph
        usetype_data = merged_data[merged_data['USETYPE'] == usetype]
        
        # Check if there's data for the selected usetype
        if usetype_data.empty:
            raise ValueError(f"No data found for USETYPE: {usetype}")

        plt.scatter(
            usetype_data['MONTHS'],
            usetype_data['Unadjusted Ratio'],
            label=usetype,
            edgecolors=colors.get(usetype, 'gray'),
            facecolors='none',
            marker=MarkerStyle("*"),
            s=20
        )

        # Draw a horizontal baseline at y = 1
        plt.axhline(y=1, color='red', linestyle='-', linewidth=1, label='Baseline')

        # Set x-axis and y-axis limits and ticks
        plt.xticks(range(0, 61, 10))  # X-axis from 0 to 60, incremented by 10
        plt.yticks(np.arange(0, 3.5, 0.5))  # Y-axis from 0 to 3, incremented by 0.5
        plt.ylim(0, 3)  # Enforce Y-axis range from 0 to 3

        # Set axis labels, title, and legend
        plt.xlabel("SalePeriod")
        plt.ylabel("Ratio")
        plt.title(f"Ratio vs SalePeriod for {usetype}")
        plt.legend(title='Use Types')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Ensure the directory exists before saving the plot
        upload_folder = "UPLOAD_FOLDER"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the combined scatter plot
        plot_image_path = os.path.join(upload_folder, f"scatter_{usetype}.png")
        plt.savefig(plot_image_path, bbox_inches='tight')
        plt.close()

        return plot_image_path

# Define file paths
sales_file_path = 'C:/Users/Harsha/Downloads/SalesExtract_v1.csv'
current_file_path = 'C:/Users/Harsha/Downloads/CurrentValueExtract_v1.csv'

# Initialize the DataProcessor class and load data
data_processor = DataProcessor(sales_file_path, current_file_path)
merged_data = data_processor.load_and_merge_data()

# Now generate a scatter plot for a specific USETYPE
usetype = 'RESIDENTIAL'  # Example usetype
plot_image_path = data_processor.generate_scatter_plot(usetype, merged_data)

print(f"Plot saved at: {plot_image_path}")
