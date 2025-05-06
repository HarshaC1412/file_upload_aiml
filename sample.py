# import os
# import logging
# import tempfile
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from flask import Flask, request, render_template, send_file
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate
# from io import BytesIO
# import chardet
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
# from reportlab.lib.styles import ParagraphStyle

# # Initialize the Flask app
# app = Flask(__name__)

# # Configuration settings
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB file size limit

# # Initialize the database and migration tool
# db = SQLAlchemy(app)
# migrate = Migrate(app, db)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the database model to hold file data
# class FileData1(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     parcel = db.Column(db.String, nullable=False)
#     tasp = db.Column(db.Float, nullable=False)
#     qualified = db.Column(db.String, nullable=False)
#     propflag = db.Column(db.String, nullable=True)

# # Detect file encoding using chardet
# def detect_encoding(file):
#     raw_data = file.read()
#     file.seek(0)  # Reset file pointer after reading
#     result = chardet.detect(raw_data)
#     return result.get('encoding', 'utf-8')  # Default to 'utf-8' if detection fails

# # Process the CSV file and save the data into the database
# def process_file(file, chunk_size=100000):
#     encoding = detect_encoding(file)
#     try:
#         temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w+b')
#         with open(temp_file.name, 'wb') as f:
#             f.write(file.read())

#         chunks = pd.read_csv(temp_file.name, encoding=encoding, chunksize=chunk_size)
#         for chunk in chunks:
#             for _, row in chunk.iterrows():
#                 file_data = FileData1(
#                     parcel=row.get('PARCEL', 'Unknown'),
#                     tasp=row.get('TASP', 0),
#                     qualified=row.get('QUALIFIED', 'Unknown'),
#                     propflag=row.get('PROPFLAG', 'N/A')
#                 )
#                 db.session.add(file_data)
#             db.session.commit()
#         temp_file.close()
#         os.remove(temp_file.name)
#     except Exception as e:
#         logging.error(f"Error processing file: {e}")
#         raise

# # Compute statistics using pandas
# def compute_statistics(df, group_col, qualified_status=None):
#     if qualified_status:
#         df = df[df['QUALIFIED_x'] == qualified_status]
#     grouped_data = df.groupby(group_col)
#     stats = {
#         group: {
#             'Mean': data['TASP_x'].mean(),
#             'Median': data['TASP_x'].median(),
#             'Variance': data['TASP_x'].var(),
#             'Standard Deviation': data['TASP_x'].std(),
#             'Mode': data['TASP_x'].mode().get(0, np.nan)
#         }
#         for group, data in grouped_data
#     }
#     return stats

# # Generate high-quality plots for the given dataframe
# def generate_high_quality_plots(df):
#     plot_buffers = []

#     pairplot_buf = BytesIO()
#     sns.pairplot(df, hue='QUALIFIED_x', diag_kind='kde')
#     plt.savefig(pairplot_buf, format='png', dpi=150)
#     plt.close()
#     pairplot_buf.seek(0)
#     plot_buffers.append(pairplot_buf)

#     boxplot_buf = BytesIO()
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x='QUALIFIED_x', y='TASP_x', data=df)
#     plt.tight_layout()
#     plt.savefig(boxplot_buf, format='png', dpi=300)
#     plt.close()
#     boxplot_buf.seek(0)
#     plot_buffers.append(boxplot_buf)

#     histplot_buf = BytesIO()
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df['TASP_x'], kde=True, bins=30)
#     plt.tight_layout()
#     plt.savefig(histplot_buf, format='png', dpi=300)
#     plt.close()
#     histplot_buf.seek(0)
#     plot_buffers.append(histplot_buf)

#     scatterplot_buf = BytesIO()
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='TASP_x', y='PROPFLAG_x', data=df)
#     plt.tight_layout()
#     plt.savefig(scatterplot_buf, format='png', dpi=300)
#     plt.close()
#     scatterplot_buf.seek(0)
#     plot_buffers.append(scatterplot_buf)

#     return plot_buffers

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         file = request.files.get('file')
#         if not file:
#             return "Please upload a CSV file.", 400

#         db.session.query(FileData1).delete()
#         db.session.commit()
#         process_file(file)

#         file_data = db.session.query(FileData1).all()
#         df = pd.DataFrame([(d.parcel, d.tasp, d.qualified, d.propflag) for d in file_data],
#                           columns=['PARCEL', 'TASP_x', 'QUALIFIED_x', 'PROPFLAG_x'])

#         stats_good_sales = compute_statistics(df, 'PROPFLAG_x', 'Q')
#         stats_bad_sales = compute_statistics(df, 'PROPFLAG_x', 'U')

#         stats_data = [{'Good Sales (Q)': stats_good_sales, 'Bad Sales (U)': stats_bad_sales}]
#         plot_buffers = generate_high_quality_plots(df)

#         pdf_output = BytesIO()
#         c = SimpleDocTemplate(pdf_output, pagesize=letter)
#         elements = [Paragraph("Statistical Report", style=ParagraphStyle(name='Title', fontSize=16)), Spacer(1, 12)]
#         for stats in stats_data:
#             elements.append(Paragraph("Statistics:", style=ParagraphStyle(name='Normal', fontSize=14)))
#             elements.append(Spacer(1, 6))
#             for group, stat in stats['Good Sales (Q)'].items():
#                 elements.append(Paragraph(f"{group}: {stat}", style=ParagraphStyle(name='Normal')))
#             elements.append(Spacer(1, 6))
#         for plot_buf in plot_buffers:
#             img = Image(plot_buf)
#             img.hAlign = 'CENTER'
#             elements.append(img)
#         c.build(elements)
#         pdf_output.seek(0)

#         return send_file(pdf_output, as_attachment=True, download_name='statistical_report.pdf', mimetype='application/pdf')
#     except Exception as e:
#         logging.error(f"Error: {e}")
#         return str(e), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# Load the two CSV files
# sales_file_path = 'C:/Users/Harsha/Downloads/SalesExtract_v1.csv'
# current_file_path = 'C:/Users/Harsha/Downloads/CurrentValueExtract_v1.csv'

# sales_data = pd.read_csv(sales_file_path)
# current_data = pd.read_csv(current_file_path)

# # Merge the data on the 'PARCEL' column
# data = pd.merge(sales_data, current_data, on='PARCEL', how='inner')

# # Handle missing values by dropping rows with missing CURRTOT or TASP values
# data = data.dropna(subset=['CURRTOT', 'TASP'])

# # Compute the Sales Ratio column
# data['SalesRatio'] = data['CURRTOT'] / data['TASP']

# # Ensure that there are no NaN or infinite values in the 'SalesRatio' column
# data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['SalesRatio'])

# # Function to compute statistical measures
# def compute_statistics(data):
#     stats = {}

#     # Median Ratio
#     stats['Median Ratio'] = data['SalesRatio'].median()

#     # Average Absolute Deviation (AAD)
#     stats['AAD'] = (data['SalesRatio'] - stats['Median Ratio']).abs().mean()

#     # Coefficient of Dispersion (COD)
#     stats['COD'] = (stats['AAD'] / stats['Median Ratio']) * 100

#     # Number of Values (N Val)
#     stats['N Val'] = len(data)

#     # Mean Ratio
#     stats['Mean Ratio'] = data['SalesRatio'].mean()

#     # Weighted Mean Ratio
#     stats['Weighted Mean Ratio'] = data['CURRTOT'].sum() / data['TASP'].sum()

#     # Standard Deviation (with safe calculation)
#     stats['Standard Deviation'] = data['SalesRatio'].std(ddof=0)

#     # Variance (with safe calculation)
#     stats['Variance'] = data['SalesRatio'].var(ddof=0)

#     # Coefficient of Variation (COV)
#     stats['COV'] = (stats['Standard Deviation'] / stats['Mean Ratio']) * 100

#     # Price-Related Differential (PRD)
#     stats['PRD'] = stats['Mean Ratio'] / stats['Weighted Mean Ratio']

#     # Price-Related Bias (PRB)
#     stats['PRB'] = (stats['Mean Ratio'] / stats['Weighted Mean Ratio']) - 1

#     return stats

# # Group by 'PROPFLAG' and 'QUALIFIED' and filter for 'Q' QUALIFIED values
# grouped_data = data.groupby(['PROPFLAG', 'QUALIFIED'])
# results_by_group = {}

# # PDF output path
# pdf_path = 'C:/Users/Harsha/Downloads/Statistical_Report.pdf'
# image_path = 'C:/Users/Harsha/Downloads/temp_plot.png'  # Temporary image file path

# # Create PDF document
# pdf = SimpleDocTemplate(pdf_path, pagesize=letter)

# # Get default styles for paragraphs
# styles = getSampleStyleSheet()

# # Content list for PDF
# content = []

# # Iterate through each group and process
# for (propflag, qualified), group in grouped_data:
#     if qualified == 'Q':
#         print(f"Processing Group: PROPFLAG={propflag}, QUALIFIED={qualified}")
        
#         # Compute statistics for the group
#         stats = compute_statistics(group)
#         results_by_group[(propflag, qualified)] = stats
        

#         # Add a title for the statistics section (using Paragraph)
#         title_text = f"Statistics for PROPFLAG={propflag}, QUALIFIED={qualified}:"
#         content.append(Paragraph(title_text, styles['Heading2']))
#         content.append(Spacer(1, 12))  # Spacer for some vertical space
        
#         # Create the data for the table
#         table_data = [['Statistic', 'Value']]  # Header row
#         for key, value in stats.items():
#             table_data.append([key, f"{value:.4f}"])

#         # Create the table
#         table = Table(table_data, colWidths=[200, 200])
#         table.setStyle(TableStyle([
#             ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#             ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#             ('FONTSIZE', (0, 0), (-1, 0), 12),
#             ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#             ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
#             ('GRID', (0, 0), (-1, -1), 1, colors.black)
#         ]))
        
#         content.append(table)  # Add table to the content
#         content.append(Spacer(1, 24))  # Spacer for vertical spacing after each section

# # Build the PDF
# pdf.build(content)

# print(f"PDF Report generated at: {pdf_path}")

# File paths
# sales_file_path = 'C:/Users/Harsha/Downloads/SalesExtract_v1.csv'
# current_file_path = 'C:/Users/Harsha/Downloads/CurrentValueExtract_v1.csv'

# # Read CSV files with error handling
# try:
#     sales_data = pd.read_csv(sales_file_path, encoding='utf-8')  # Adjust encoding if needed
#     current_data = pd.read_csv(current_file_path, encoding='utf-8')
# except Exception as e:
#     print("Error reading CSV files:", e)
#     exit()

# # Merge data
# merged_data = pd.concat([sales_data, current_data], ignore_index=True)

# # Clean column names (remove spaces)
# merged_data.columns = merged_data.columns.str.strip()

# # Ensure required columns exist
# required_columns = {'QUALIFIED', 'PROPFLAG', 'SYEAR', 'SMONTH', 'TASP', 'CURRTOT'}
# missing_columns = required_columns - set(merged_data.columns)
# if missing_columns:
#     print(f"Error: Missing columns: {missing_columns}")
#     exit()

# # Convert columns to proper format
# merged_data['QUALIFIED'] = merged_data['QUALIFIED'].astype(str).str.strip()
# merged_data['PROPFLAG'] = merged_data['PROPFLAG'].astype(str).str.strip().str.upper()

# # Debugging: Check unique values before filtering
# print("Unique QUALIFIED values:", merged_data['QUALIFIED'].unique())
# print("Unique PROPFLAG values:", merged_data['PROPFLAG'].unique())

# # Filter for QUALIFIED == 'Q' and PROPFLAG == 'RESIDENTIAL'
# filtered_data = merged_data[
#     (merged_data['QUALIFIED'] == 'Q') &
#     (merged_data['PROPFLAG'] == 'RESIDENTIAL')
# ]

# # Debugging: Check if data exists after filtering
# if filtered_data.empty:
#     print("No data found for QUALIFIED='Q' and PROPFLAG='RESIDENTIAL'. Exiting.")
#     exit()

# print("Filtered Data Sample:")
# print(filtered_data.head())

# # Calculate 'Unadjusted Ratio'
# filtered_data['Unadjusted Ratio'] = filtered_data['TASP'] / filtered_data['CURRTOT']

# # Remove infinities and NaN values
# filtered_data = filtered_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['SYEAR', 'SMONTH', 'Unadjusted Ratio'])

# # Compute base year and base month
# base_year = filtered_data['SYEAR'].min()
# base_month = filtered_data.loc[filtered_data['SYEAR'] == base_year, 'SMONTH'].min()

# # Compute 'MONTHS' column
# filtered_data['MONTHS'] = ((filtered_data['SYEAR'] - base_year) * 12) + (filtered_data['SMONTH'] - base_month)

# # Normalize 'MONTHS' to start from 0
# filtered_data['MONTHS'] = filtered_data['MONTHS'] - filtered_data['MONTHS'].min()

# # Sort by 'MONTHS' column
# filtered_data = filtered_data.sort_values(by='MONTHS', ascending=True)

# # Debugging: Check final data before plotting
# print("Final Data Sample for Plotting:")
# print(filtered_data[['MONTHS', 'Unadjusted Ratio']].head())

# # Scatter plot
# plt.figure(figsize=(12, 6))
# plt.scatter(
#     filtered_data['MONTHS'], 
#     filtered_data['Unadjusted Ratio'], 
#     color='green', 
#     alpha=0.6, 
#     edgecolors='black'
# )

# # Draw a baseline at y=1
# plt.axhline(y=1, color='red', linestyle='-', linewidth=1, label="Baseline")

# # Set x-axis and y-axis limits and ticks
# plt.xticks(range(0, 61, 10))  # X-axis from 0 to 60, incremented by 10
# plt.yticks(np.arange(0, 3.5, 0.5))  # Y-axis from 0 to 3, incremented by 0.5
# plt.ylim(0, 3)  # Enforce Y-axis range from 0 to 3

# # Set axis labels, title, and legend
# plt.xlabel("SalePeriod")
# plt.ylabel("Ratio")
# plt.title("Ratio vs SalePeriod for RESIDENTIAL")
# plt.legend(title='Use Types')
# plt.grid(True, linestyle='--', alpha=0.7)

# # Show the plot
# plt.show()

sales_file_path = 'C:/Users/Harsha/Downloads/SalesCV_join.csv'
sales_data = pd.read_csv(sales_file_path)
print(type(sales_data['PROPFLAG']))
filtered_data = sales_data[(sales_data['QUALIFIED'] == 'Q') & (sales_data['UseType'] == 'VACANT LAND')]
plt.figure(figsize=(12, 6))
plt.scatter(
    filtered_data['MONTHS'], 
    filtered_data['RATIO'], 
    color='green', 
    facecolors='none',
    s=20,
    marker=MarkerStyle('*'),
    alpha=0.6, 
    edgecolors='black'
)

# Draw a baseline at y=1
plt.axhline(y=1, color='red', linestyle='-', linewidth=1, label="Baseline")

# Set x-axis and y-axis limits and ticks
plt.xticks(range(0, 61, 10))  # X-axis from 0 to 60, incremented by 10
plt.yticks(np.arange(0, 3.5, 0.5))  # Y-axis from 0 to 3, incremented by 0.5
plt.ylim(0, 3)  # Enforce Y-axis range from 0 to 3

# Set axis labels, title, and legend
plt.xlabel("SalePeriod")
plt.ylabel("Ratio")
plt.title("Ratio vs SalePeriod for RESIDENTIAL")
plt.legend(title='Use Types')
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()