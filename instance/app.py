from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, flash, send_file, session,abort
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image
from werkzeug.utils import secure_filename
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
import uuid
from base64 import b64encode
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data1.db'
app.config['SECRET_KEY'] = 'Indi@1947'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Database setup
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)

# Upload and result folders
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

# Forms
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user = User.query.filter_by(username=username.data).first()
        if existing_user:
            raise ValidationError("Username already exists. Please choose a different one.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Incorrect username or password', 'danger')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/validate_data', methods=['POST'])
def validate_data():
    files = request.files
    uploaded_files = {}

    for file_key in files:
        file = files[file_key]
        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            uploaded_files[file_key] = file_path

    if not uploaded_files:
        return jsonify({"status": "error", "message": "No valid CSV files uploaded"})

    # Basic validation: Check if all required files are uploaded
    required_files = ['l_file', 's_file', 'c_file', 'h_file', 'i_file']
    for file in required_files:
        if file not in uploaded_files:
            return jsonify({"status": "error", "message": f"Missing required file: {file}"})

    return jsonify({"status": "success", "message": "Files uploaded and validated successfully"})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/results', methods=['POST'])
def results():
    files = {
        "land": request.files.get("l_file"),
        "sales": request.files.get("s_file"),
        "current": request.files.get("c_file"),
        "historical": request.files.get("h_file"),
        "improvement": request.files.get("i_file"),
    }

    file_paths = {}
    for key, file in files.items():
        if file and file.filename.endswith(".csv"):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            file_paths[key] = file_path
    
    # Process the data and get the df_sales DataFrame
    df_sales = process_data(file_paths)
    
    # Debug: Check the processed data
    print("Processed DataFrame:")
    print(df_sales.head())
    print(df_sales.info())

    # Generate a unique file ID and save the DataFrame to a file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    df_sales.to_json(file_path, orient='split')
    
    # Generate PDF report
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.pdf")
    generate_pdf_report(file_id)
    
    # Redirect to the report route with the file ID
    return redirect(url_for('report', file_id=file_id))

def assign_usetype(abstrprd):
    if 0 <= abstrprd <= 999:
        return 'VACANT LAND'
    elif 1000 <= abstrprd <= 1999:
        return 'RESIDENTIAL'
    elif 2000 <= abstrprd <= 2999:
        return 'COMMERCIAL'
    elif 3000 <= abstrprd <= 3999:
        return 'INDUSTRIAL'
    elif 4000 <= abstrprd <= 4999:
        return 'AGRICULTURE'
    elif 9000 <= abstrprd <= 9999:
        return 'EXEMPT'
    return 'OTHER'

def process_data(file_paths):
    try:
        # Step 1: Load datasets
        df_land = pd.read_csv(file_paths['land'])
        df_sales = pd.read_csv(file_paths['sales'])
        df_current_value = pd.read_csv(file_paths['current'])
        df_historical = pd.read_csv(file_paths['historical'])

        # Step 2: Identify duplicate parcels in LandFile (load in array1)
        duplicates = df_land[df_land.duplicated('PARCEL', keep=False)]
        array1 = duplicates['PARCEL'].unique()

        # Step 3: Remove duplicate parcels (from array1) from SalesFile
        df_sales = df_sales[~df_sales['PARCEL'].isin(array1)]

        # Step 4: Remove duplicate parcels (from array1) from LandFile
        df_land = df_land[~df_land['PARCEL'].isin(array1)]

        # Step 5: Find PARCELS in SalesFile whose ABSTRPRD does not match ABSTRPRD in LandFile (load in another array2)
        mismatched_parcels = df_sales.merge(df_land[['PARCEL', 'ABSTRPRD']], on='PARCEL', how='left')
        mismatched_parcels = mismatched_parcels[mismatched_parcels['ABSTRPRD_x'] != mismatched_parcels['ABSTRPRD_y']]
        array2 = mismatched_parcels['PARCEL'].unique()

        # Step 6: Delete these non-matching ABSTRPRD parcels (in array2) from both SalesFile and LandFile
        df_sales = df_sales[~df_sales['PARCEL'].isin(array2)]
        df_land = df_land[~df_land['PARCEL'].isin(array2)]

        # Step 7: Merge LandFile with CurrentValue on PARCEL field and bring over ABSTRPRD to CurrentValue
        df_current_value = df_current_value.merge(df_land[['PARCEL', 'ABSTRPRD']], on='PARCEL', how='left', suffixes=('', '_land'))

        # Step 8: Merge HistoricalFile with CurrentValue on 'PARCEL' field and bring over PREVTOT to CurrentValue
        df_current_value = df_current_value.merge(df_historical[['PARCEL', 'PREVTOT']], on='PARCEL', how='left', suffixes=('', '_hist'))

        # Step 9: Compute the percentage change (PCTCHANGE with 2 decimals) between PREVTOT and CURRTOT in CurrentValue
        df_current_value['CURRTOT'] = df_current_value.get('CURRTOT', 0)  # Ensure CURRTOT exists
        df_current_value['PREVTOT'] = df_current_value.get('PREVTOT', 0)  # Ensure PREVTOT exists
        df_current_value[['CURRTOT', 'PREVTOT']] = df_current_value[['CURRTOT', 'PREVTOT']].fillna(0)  # Fill NaN with 0
        df_current_value['PCTCHANGE'] = np.where(
            df_current_value['PREVTOT'] == 0, 0,  # Avoid division by zero
            ((df_current_value['CURRTOT'] - df_current_value['PREVTOT']) / df_current_value['PREVTOT']) * 100
        )
        df_current_value['PCTCHANGE'] = df_current_value['PCTCHANGE'].round(2)  # Round to 2 decimal places

        # Step 10: Merge SalesFile with CurrentValue on PARCEL field and bring over CURRTOT and PCTCHANGE to SalesFile
        df_sales = df_sales.merge(df_current_value[['PARCEL', 'CURRTOT', 'PCTCHANGE']], on='PARCEL', how='left', suffixes=('', '_cv'))

        # Step 11: Add and Compute RATIO field
        if 'CURRTOT' not in df_sales.columns:
            df_sales['CURRTOT'] = 0  # Default value if column is missing
        if 'TASP' not in df_sales.columns:
            df_sales['TASP'] = 0  # Default value if column is missing
        df_sales[['CURRTOT', 'TASP']] = df_sales[['CURRTOT', 'TASP']].fillna(0)  # Fill NaN with 0
        df_sales['RATIO'] = np.where(
            df_sales['TASP'] == 0, 0,  # Avoid division by zero
            df_sales['CURRTOT'] / df_sales['TASP']
        )
        df_sales['RATIO'] = df_sales['RATIO'].round(2)  # Round to 2 decimal places

        df_sales['USETYPE'] = df_sales['ABSTRPRD'].apply(assign_usetype)
        df_sales = df_sales[df_sales['USETYPE'].isin(['RESIDENTIAL', 'COMMERCIAL', 'VACANT LAND'])]
        df_sales = df_sales[df_sales['QUALIFIED'] == 'Q']

        # Ensure all required columns exist
        required_columns = ['SYEAR', 'RATIO', 'CURRTOT', 'TASP', 'PCTCHANGE', 'USETYPE']
        for col in required_columns:
            if col not in df_sales.columns:
                df_sales[col] = 0  # Add default value if column is missing

        print(df_sales.head())  # Debugging: Print the first few rows of the processed data
        return df_sales
    except Exception as e:
        print(f"Error processing data: {e}")
        return pd.DataFrame()

def compute_statistics(group):
    return {
        'SYEAR': group['SYEAR'].iloc[0],
        'Median Ratio': round(group['RATIO'].median(), 2),
        'COD': round(((group['RATIO'] - group['RATIO'].median()).abs().mean() / group['RATIO'].median()) * 100, 2),
        'N': len(group),
        'Mean Ratio': round(group['RATIO'].mean(), 2),
        'Weighted Mean Ratio': round(group['CURRTOT'].sum() / group['TASP'].sum(), 2),
        'PRD': round(group['RATIO'].mean() / (group['CURRTOT'].sum() / group['TASP'].sum()), 2),
    }

def compute_pctchange_stats(group):
    return {
        'SYEAR': group['SYEAR'].iloc[0],
        'N': len(group),
        'Mean': round(group['PCTCHANGE'].mean(), 2),
        'Median': round(group['PCTCHANGE'].median(), 2),
        'Min': round(group['PCTCHANGE'].min(), 2),
        'Max': round(group['PCTCHANGE'].max(), 2)
    }

# Function to calculate the total row for table1_data
def calculate_total_table1(data):
    total_N = 0
    total_Median_Ratio = 0
    total_COD = 0
    total_Mean_Ratio = 0
    total_Weighted_Mean_Ratio = 0
    total_PRD = 0
    count = len(data) - 1  # Subtract 1 to exclude the header row

    for row in data[1:]:  # Skip the header row
        total_N += row[3]  # N is at index 3
        total_Median_Ratio += row[1]  # Median Ratio is at index 1
        total_COD += row[2]  # COD is at index 2
        total_Mean_Ratio += row[4]  # Mean Ratio is at index 4
        total_Weighted_Mean_Ratio += row[5]  # Weighted Mean Ratio is at index 5
        total_PRD += row[6]  # PRD is at index 6

    # Calculate averages and round to 2-3 decimal places
    avg_Median_Ratio = round(total_Median_Ratio / count, 3)
    avg_COD = round(total_COD / count, 3)
    avg_Mean_Ratio = round(total_Mean_Ratio / count, 3)
    avg_Weighted_Mean_Ratio = round(total_Weighted_Mean_Ratio / count, 3)
    avg_PRD = round(total_PRD / count, 3)

    # Return the total row
    return ["Total", avg_Median_Ratio, avg_COD, total_N, avg_Mean_Ratio, avg_Weighted_Mean_Ratio, avg_PRD]

# Function to calculate the total row for table2_data
def calculate_total_table2(data):
    total_N = 0
    total_Mean = 0
    total_Median = 0
    total_Min = 0
    total_Max = 0
    count = len(data) - 1  # Subtract 1 to exclude the header row

    for row in data[1:]:  # Skip the header row
        total_N += row[1]  # N is at index 1
        total_Mean += row[2]  # Mean is at index 2
        total_Median += row[3]  # Median is at index 3
        total_Min += row[4]  # Min is at index 4
        total_Max += row[5]  # Max is at index 5

    # Calculate averages and round to 2-3 decimal places
    avg_Mean = round(total_Mean / count, 3)
    avg_Median = round(total_Median / count, 3)
    avg_Min = round(total_Min / count, 3)
    avg_Max = round(total_Max / count, 3)

    # Return the total row
    return ["Total", total_N, avg_Mean, avg_Median, avg_Min, avg_Max]


def generate_pdf_report(file_id):
    pdf_path = os.path.join("static", f"report_{file_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Audit Ratios", styles["Title"]))
    elements.append(Spacer(1, 12))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    if not os.path.exists(file_path):
        return "Data file not found."

    df_sales = pd.read_json(file_path, orient='split')
    required_columns = {'SYEAR', 'SMONTH', 'RATIO', 'CURRTOT', 'TASP', 'PCTCHANGE', 'USETYPE'}
    for col in required_columns:
        if col not in df_sales.columns:
            df_sales[col] = 0  # Default value if column is missing

    usetypes = df_sales['USETYPE'].unique()
    table1_data = {usetype: [compute_statistics(group) for _, group in df_sales[df_sales['USETYPE'] == usetype].groupby('SYEAR')] for usetype in usetypes}
    table2_data = {usetype: [compute_pctchange_stats(group) for _, group in df_sales[df_sales['USETYPE'] == usetype].groupby('SYEAR')] for usetype in usetypes}
    summary_data = df_sales.groupby('USETYPE')['PCTCHANGE'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index().round(2).to_dict('records')

    # Calculate months for scatter plot
    base_year = df_sales['SYEAR'].min()
    base_month = df_sales.loc[df_sales['SYEAR'] == base_year, 'SMONTH'].min()
    df_sales['MONTHS'] = ((df_sales['SYEAR'] - base_year) * 12) + (df_sales['SMONTH'] - base_month)
    df_sales['MONTHS'] = df_sales['MONTHS'] - df_sales['MONTHS'].min()

    colors_map = {'RESIDENTIAL': 'green', 'COMMERCIAL': 'red', 'VACANT LAND': 'blue'}
    pastel_colors = [colors.lightblue, colors.lavender, colors.mintcream, colors.lightpink, colors.honeydew]

    for idx, usetype in enumerate(usetypes):
        elements.append(Paragraph(f"Ratios for {usetype}", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # Generate scatter plot
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        plot_path = f"C:/Users/Harsha/Downloads/{usetype}_scatter.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(
            filtered_data['MONTHS'], filtered_data['RATIO'], 
            alpha=0.5, color=colors_map.get(usetype, 'black'), 
            marker='x', s=20
        )
        plt.xlabel('Months')
        plt.ylabel('Ratio')
        plt.ylim(0, 3)
        plt.xlim(0, 60)
        plt.title(f'Scatter Plot for {usetype}')
        plt.savefig(plot_path)
        plt.close()

        data = [["SYEAR", "Median Ratio", "COD", "N", "Mean Ratio", "Weighted Mean Ratio", "PRD"]]
        for row in table1_data.get(usetype, []):
            data.append([row['SYEAR'], row['Median Ratio'], row['COD'], row['N'], 
                        row['Mean Ratio'], row['Weighted Mean Ratio'], row['PRD']])

        # Calculate and append total row
        total_row = calculate_total_table1(data)
        data.append(total_row)

        table = Table(data, colWidths=[70, 90, 60, 60, 90, 120, 60])  # Adjusted column widths
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pastel_colors[0]),  # Header background
            ('BACKGROUND', (0, -1), (-1, -1), pastel_colors[1]),  # Total row background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header bold
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Total row bold
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -2), pastel_colors[2]),  # Body background
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Percentage Change Stats for {usetype}", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # Second Table
        data2 = [["SYEAR", "N", "Mean", "Median", "Min", "Max"]]
        for row in table2_data.get(usetype, []):
            data2.append([row['SYEAR'], row['N'], row['Mean'], row['Median'], 
                        row['Min'], row['Max']])

        # Calculate and append total row
        total_row2 = calculate_total_table2(data2)
        data2.append(total_row2)

        table2 = Table(data2, colWidths=[70, 60, 80, 80, 80, 80])  # Adjusted column widths
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pastel_colors[0]),  # Header background
            ('BACKGROUND', (0, -1), (-1, -1), pastel_colors[1]),  # Total row background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header bold
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),  # Total row bold
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -2), pastel_colors[2]),  # Body background
        ]))
        elements.append(table2)
        elements.append(Spacer(1, 12))

        # Add scatter plot image
        if os.path.exists(plot_path):
            elements.append(Image(plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))

    # Summary Table
    elements.append(Paragraph("Unsold Properties", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    data = [["USETYPE", "N", "Mean", "Median", "Min", "Max"]]
    for row in summary_data:
        data.append([row['USETYPE'], row['count'], row['mean'], row['median'], row['min'], row['max']])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)

    doc.build(elements)
    return pdf_path


@app.route('/report')
def report():
    file_id = request.args.get('file_id')
    if not file_id:
        return "No data found. Please process the data first.", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    if not os.path.exists(file_path):
        return "Data file not found.", 404

    df_sales = pd.read_json(file_path, orient='split')
    required_columns = {'SYEAR', 'RATIO', 'CURRTOT', 'TASP', 'PCTCHANGE', 'USETYPE'}
    for col in required_columns:
        if col not in df_sales.columns:
            df_sales[col] = 0  # Default value if column is missing

    usetypes = df_sales['USETYPE'].unique()
    table1_data = {usetype: [compute_statistics(group) for _, group in df_sales[df_sales['USETYPE'] == usetype].groupby('SYEAR')] for usetype in usetypes}
    table2_data = {usetype: [compute_pctchange_stats(group) for _, group in df_sales[df_sales['USETYPE'] == usetype].groupby('SYEAR')] for usetype in usetypes}
    summary_data = df_sales.groupby('USETYPE')['PCTCHANGE'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index().round(2).to_dict('records')

    base_year = df_sales['SYEAR'].min()
    df_sales['MONTHS'] = (df_sales['SYEAR'] - base_year) * 12

    colors_map = {'RESIDENTIAL': 'green', 'COMMERCIAL': 'red', 'VACANT LAND': 'blue'}
    os.makedirs("static", exist_ok=True)

    for usetype in usetypes:
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        plot_path = os.path.join("static", f"{usetype}_scatter.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(filtered_data['MONTHS'], filtered_data['RATIO'], alpha=0.5, color=colors_map.get(usetype, 'black'), marker='x', s=20)
        plt.xlabel('Months')
        plt.ylabel('Ratio')
        plt.ylim(0, 3)
        plt.xlim(0, 60)
        plt.title(f'Scatter Plot for {usetype}')
        plt.savefig(plot_path)
        plt.close()

    pdf_path = os.path.join("static", "report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Sales Report", styles["Title"]))
    pastel_colors = [colors.HexColor("#FFB3BA"), colors.HexColor("#FFDFBA"), colors.HexColor("#FFFFBA"), colors.HexColor("#BAFFC9"), colors.HexColor("#BAE1FF")]

    for idx, (usetype, stats) in enumerate(table1_data.items()):
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Statistics for {usetype}", styles["Heading2"]))
        data = [["SYear", "Median Ratio", "COD", "N", "Mean Ratio", "Weighted Mean Ratio", "PRD"]]
        data.extend([[
            year_stats.get('SYEAR', ''),
            round(year_stats.get('Median Ratio', 0), 2),
            round(year_stats.get('COD', 0), 2),
            year_stats.get('N', 0),
            round(year_stats.get('Mean Ratio', 0), 2),
            round(year_stats.get('Weighted Mean Ratio', 0), 2),
            round(year_stats.get('PRD', 0), 2)
        ] for year_stats in stats])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), pastel_colors[idx % len(pastel_colors)]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Image(os.path.join("static", f"{usetype}_scatter.png")))

    doc.build(elements)
    return render_template('report.html', usetypes=usetypes, table1_data=table1_data, table2_data=table2_data, summary_data=summary_data, download_link=f"/download_pdf?file_id={file_id}")


@app.route('/download_pdf')
def download_pdf():
    file_id = request.args.get('file_id')
    if not file_id:
        return "No file ID provided.", 400

    pdf_path = os.path.join("static", f"report_{file_id}.pdf")
    if not os.path.exists(pdf_path):
        return abort(404, description="Requested PDF file not found.")

    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)