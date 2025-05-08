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
from sklearn.linear_model import LinearRegression
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image,PageBreak
from reportlab.lib.enums import TA_CENTER
from werkzeug.utils import secure_filename
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import statsmodels.api as sm
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

    # Define required columns for each file with updated specifications
    required_columns = {
        'l_file': ['PARCEL', 'ABSTRLND', 'LNDSIZE'],  # Updated Land file columns
        's_file': ['PARCEL', 'ABSTRPRD', 'NETPRICE', 'TASP'],  # Updated Sales file columns
        'c_file': ['PARCEL', 'CURRTOT'],  # Updated Current Value file columns
        'h_file': ['PARCEL', 'PREVTOT'],  # Updated Historical file columns
        'i_file': ['PARCEL', 'ABSTRIMP', 'LIVEAREA', 'EFFBLT', 'CONDITION', 'QUALITY', 'ARCSTYLE', 'FINBSMNT', 'GARTYPE', 'BASEMENT']  # Updated Improvement file columns
    }

    # Mapping of file keys to display names
    file_display_names = {
        'l_file': 'Land File',
        's_file': 'Sales File',
        'c_file': 'Current Value File',
        'h_file': 'Historical File',
        'i_file': 'Improvement File'
    }

    validation_messages = {}
    has_errors = False  # Flag to track if there are any validation errors

    for file_key in files:
        file = files[file_key]
        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            uploaded_files[file_key] = file_path

            # Read the CSV file to check columns
            try:
                df = pd.read_csv(file_path)
                # Adjust required columns if SUBCLASS is present instead
                check_columns = required_columns[file_key]
                if file_key in ['l_file', 's_file', 'i_file'] and 'SUBCLASS' in df.columns:
                    if file_key == 'l_file' and 'ABSTRLND' not in df.columns:
                        check_columns = ['PARCEL', 'SUBCLASS', 'LNDSIZE']
                    elif file_key == 's_file' and 'ABSTRPRD' not in df.columns:
                        check_columns = ['PARCEL', 'SUBCLASS', 'NETPRICE', 'TASP']
                    elif file_key == 'i_file' and 'ABSTRIMP' not in df.columns:
                        check_columns = ['PARCEL', 'SUBCLASS', 'LIVEAREA', 'EFFBLT', 'CONDITION', 'QUALITY', 'ARCSTYLE', 'FINBSMNT', 'GARTYPE', 'BASEMENT']

                missing_columns = [col for col in check_columns if col not in df.columns]
                if missing_columns:
                    validation_messages[file_key] = f'<span style="color: red;font-size: 20px;"><strong>Missing columns: {", ".join(missing_columns)}</strong></span>'
                    has_errors = True  # Set flag to True if there are missing columns
                else:
                    # Include the list of required columns in the success message
                    validation_messages[file_key] = (
                        f"All required columns are present: {', '.join(check_columns)}"
                    )
            except Exception as e:
                validation_messages[file_key] = f"Error reading file: {str(e)}"
                has_errors = True  # Set flag to True if there is an error reading the file

    if not uploaded_files:
        return jsonify({"status": "error", "message": "No valid CSV files uploaded"})

    # Basic validation: Check if all required files are uploaded
    required_files = ['l_file', 's_file', 'c_file', 'h_file', 'i_file']
    for file in required_files:
        if file not in uploaded_files:
            return jsonify({"status": "error", "message": f"Missing required file: {file_display_names[file]}"})

    # Replace file keys with display names in validation messages
    formatted_validation_messages = {
        file_display_names[file_key]: message
        for file_key, message in validation_messages.items()
    }

    # Determine the overall status based on whether there are any errors
    status = "error" if has_errors else "success"

    # Return validation messages for each file
    return jsonify({
        "status": status,
        "message": "Files uploaded and validated successfully" if not has_errors else "Validation errors found",
        "validation_messages": formatted_validation_messages
    })

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

    # Get month filters from the form
    month_filters = {
        "RESIDENTIAL": int(request.form.get("res", 0)),  # Default to 0 if not provided
        "COMMERCIAL": int(request.form.get("com", 0)),
        "VACANT LAND": int(request.form.get("vl", 0)),
    }

    county_name = request.form.get('cname')  

    file_paths = {}
    for key, file in files.items():
        if file and file.filename.endswith(".csv"):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            file_paths[key] = file_path

    # Process the data with month filters
    df_sales, df_current_value, bldgnum_1_array = process_data(file_paths, month_filters)

    # Generate a unique file ID and save both DataFrames
    file_id = str(uuid.uuid4())
    sales_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"current_{file_id}.json")
    
    df_sales.to_json(sales_file_path, orient='split')
    df_current_value.to_json(current_file_path, orient='split')

    # Generate PDF report with month filters
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.pdf")
    generate_pdf_report(file_id, month_filters, county_name)

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

def process_data(file_paths, month_filters):
    try:
        # Step 1: Load datasets
        df_land = pd.read_csv(file_paths['land'])
        df_sales = pd.read_csv(file_paths['sales'])
        df_current_value = pd.read_csv(file_paths['current'])
        df_historical = pd.read_csv(file_paths['historical'])
        df_improvement = pd.read_csv(file_paths['improvement'])

        # Step 2: Join sales and improvement immediately after loading
        # Normalize PARCEL for consistent merging
        df_sales['PARCEL'] = df_sales['PARCEL'].astype(str)
        df_improvement['PARCEL'] = df_improvement['PARCEL'].astype(str)

        # Step 3: Rename ABSTRIMP or SUBCLASS to ABSTRPRD in df_improvement
        if 'ABSTRIMP' in df_improvement.columns:
            df_improvement.rename(columns={'ABSTRIMP': 'ABSTRPRD'}, inplace=True)
            print("Renamed ABSTRIMP to ABSTRPRD in df_improvement")
        elif 'SUBCLASS' in df_improvement.columns:
            df_improvement.rename(columns={'SUBCLASS': 'ABSTRPRD'}, inplace=True)
            print("Renamed SUBCLASS to ABSTRPRD in df_improvement")
        else:
            print("Warning: Neither 'ABSTRIMP' nor 'SUBCLASS' found in df_improvement")

        # Step 4: Filter df_improvement for BLDGNUM = '1' and store in array
        if 'BLDGNUM' in df_improvement.columns:
            df_improvement['BLDGNUM'] = df_improvement['BLDGNUM']
            df_improvement = df_improvement[df_improvement['BLDGNUM'] == 1]
            bldgnum_1_array = df_improvement.to_numpy()
            print(f"Rows after BLDGNUM filter: {len(df_improvement)}")
        else:
            print("Warning: 'BLDGNUM' column not found in improvement file")
            bldgnum_1_array = np.array([])
            df_improvement = pd.DataFrame(columns=['PARCEL', 'EFFBLT', 'ARCSTYLE', 'ABSTRPRD', 'GARTYPE', 'FINBSMNT', 'BASEMENT'])

        # Step 5: Categorize residential based on ABSTRPRD in df_improvement
        if not df_improvement.empty:
            if 'ABSTRPRD' in df_improvement.columns:
                df_improvement['USETYPE'] = df_improvement['ABSTRPRD'].apply(assign_usetype)
                df_improvement_residential = df_improvement[df_improvement['USETYPE'] == 'RESIDENTIAL']
                print(f"Rows after residential filter in df_improvement (based on ABSTRPRD): {len(df_improvement_residential)}")
            else:
                print("Error: 'ABSTRPRD' column missing in df_improvement after renaming; cannot categorize residential")
                df_improvement_residential = pd.DataFrame(columns=['PARCEL', 'EFFBLT', 'ARCSTYLE', 'GARTYPE', 'FINBSMNT', 'BASEMENT'])
        else:
            print("Warning: df_improvement is empty after BLDGNUM filter")
            df_improvement_residential = pd.DataFrame(columns=['PARCEL', 'EFFBLT', 'ARCSTYLE', 'GARTYPE', 'FINBSMNT', 'BASEMENT'])

        # Step 6: Merge df_sales with df_improvement_residential
        print("Sample PARCELs in df_sales before merge:", df_sales['PARCEL'].head().tolist())
        print("Sample PARCELs in df_improvement_residential:", df_improvement_residential['PARCEL'].head().tolist())

        # Perform the merge
        df_sales = df_sales.merge(df_improvement_residential[['PARCEL', 'EFFBLT', 'ARCSTYLE', 'CONDITION', 'QUALITY', 'LIVEAREA', 'GARTYPE', 'FINBSMNT', 'BASEMENT']],
                                  on='PARCEL', how='left')
        print("df_sales head after initial merge with improvement:")
        print(df_sales[['PARCEL', 'EFFBLT', 'ARCSTYLE', 'GARTYPE', 'FINBSMNT', 'BASEMENT']].head())

        # Step 7: Rename ABSTRLND or SUBCLASS to ABSTRPRD in df_land
        if 'ABSTRLND' in df_land.columns:
            df_land.rename(columns={'ABSTRLND': 'ABSTRPRD'}, inplace=True)
            print("Renamed ABSTRLND to ABSTRPRD in df_land")
        elif 'SUBCLASS' in df_land.columns:
            df_land.rename(columns={'SUBCLASS': 'ABSTRPRD'}, inplace=True)
            print("Renamed SUBCLASS to ABSTRPRD in df_land")
        else:
            print("Warning: Neither 'ABSTRLND' nor 'SUBCLASS' found in df_land")

        # Step 8: Rename SUBCLASS to ABSTRPRD in df_sales if necessary
        if 'SUBCLASS' in df_sales.columns and 'ABSTRPRD' not in df_sales.columns:
            df_sales.rename(columns={'SUBCLASS': 'ABSTRPRD'}, inplace=True)
            print("Renamed SUBCLASS to ABSTRPRD in df_sales")

        # Step 9: Duplicate and mismatch removal
        duplicates = df_land[df_land.duplicated('PARCEL', keep=False)]
        array1 = duplicates['PARCEL'].unique()
        df_sales = df_sales[~df_sales['PARCEL'].isin(array1)]
        df_land = df_land[~df_land['PARCEL'].isin(array1)]

        mismatched_parcels = df_sales.merge(df_land[['PARCEL', 'ABSTRPRD', 'LNDSIZE']], on='PARCEL', how='left')
        mismatched_parcels = mismatched_parcels[mismatched_parcels['ABSTRPRD_x'] != mismatched_parcels['ABSTRPRD_y']]
        array2 = mismatched_parcels['PARCEL'].unique()
        df_sales = df_sales[~df_sales['PARCEL'].isin(array2)]
        df_land = df_land[~df_land['PARCEL'].isin(array2)]

        # Merge LNDSIZE into df_sales
        df_sales = df_sales.merge(df_land[['PARCEL', 'LNDSIZE']], on='PARCEL', how='left')
        print("df_sales head after merging LNDSIZE from df_land:")
        print(df_sales[['PARCEL', 'LNDSIZE']].head())

        # Step 10: Merging and computations
        df_current_value = df_current_value.merge(df_land[['PARCEL', 'ABSTRPRD']], on='PARCEL', how='left', suffixes=('', '_land'))
        df_current_value = df_current_value.merge(df_historical[['PARCEL', 'PREVTOT']], on='PARCEL', how='left', suffixes=('', '_hist'))
        df_current_value['CURRTOT'] = df_current_value.get('CURRTOT', 0).fillna(0)
        df_current_value['PREVTOT'] = df_current_value.get('PREVTOT', 0).fillna(0)
        df_current_value['PCTCHANGE'] = np.where(
            df_current_value['PREVTOT'] == 0, 0,
            ((df_current_value['CURRTOT'] - df_current_value['PREVTOT']) / df_current_value['PREVTOT'])
        ).round(4)

        df_sales = df_sales.merge(df_current_value[['PARCEL', 'CURRTOT', 'PCTCHANGE']], on='PARCEL', how='left', suffixes=('', '_cv'))
        df_sales['RATIO'] = np.where(
            df_sales['TASP'] == 0, 0,
            df_sales['CURRTOT'] / df_sales['TASP']
        ).round(2)

        # Step 11: Assign USETYPE and initial filtering
        if 'ABSTRPRD' not in df_sales.columns:
            print("Error: 'ABSTRPRD' column missing in df_sales")
            return pd.DataFrame(), df_current_value, bldgnum_1_array
        df_sales['USETYPE'] = df_sales['ABSTRPRD'].apply(assign_usetype)
        df_sales = df_sales[df_sales['USETYPE'].isin(['RESIDENTIAL', 'COMMERCIAL', 'VACANT LAND'])]
        if 'QUALIFIED' in df_sales.columns:
            df_sales = df_sales[df_sales['QUALIFIED'] == 'Q']
        else:
            print("Warning: 'QUALIFIED' column not found in df_sales; skipping filter")

        # Step 12: Reorder USETYPE
        df_sales = df_sales.sort_values(by='USETYPE', key=lambda x: x.map({'RESIDENTIAL': 0, 'COMMERCIAL': 1, 'VACANT LAND': 2}))

        # Step 13: Calculate MONTHS
        if 'SYEAR' not in df_sales.columns:
            print("Error: 'SYEAR' column missing in df_sales")
            return pd.DataFrame(), df_current_value, bldgnum_1_array
        base_year = df_sales['SYEAR'].max()
        if 'SMONTH' in df_sales.columns:
            base_month = df_sales.loc[df_sales['SYEAR'] == base_year, 'SMONTH'].max()
            df_sales['MONTHS'] = ((base_year - df_sales['SYEAR']) * 12) + (base_month - df_sales['SMONTH'])
        else:
            df_sales['MONTHS'] = (base_year - df_sales['SYEAR']) * 12

        # Step 14: Filter by month_filters
        filtered_data = pd.DataFrame()
        for usetype, max_months in month_filters.items():
            usetype_data = df_sales[df_sales['USETYPE'] == usetype]
            if not usetype_data.empty:
                usetype_data_filtered = usetype_data[usetype_data['MONTHS'] <= max_months]
                filtered_data = pd.concat([filtered_data, usetype_data_filtered])
        df_sales = filtered_data

        # Step 15: Calculate UnadjRatio = CURRTOT / NETPRICE
        if 'NETPRICE' in df_sales.columns and 'CURRTOT' in df_sales.columns:
            df_sales['UnadjRatio'] = np.where(
                df_sales['NETPRICE'] == 0, 0,
                (df_sales['CURRTOT'] / df_sales['NETPRICE']).round(2)
            )
        else:
            print("Warning: Cannot calculate UnadjRatio - 'NETPRICE' or 'CURRTOT' column missing")
            df_sales['UnadjRatio'] = np.nan

        # Step 16: Ensure all required columns exist
        required_columns = ['SYEAR', 'RATIO', 'CURRTOT', 'TASP', 'PCTCHANGE', 'USETYPE', 'MONTHS', 'EFFBLT', 'ARCSTYLE', 'CONDITION', 'QUALITY', 'LIVEAREA', 'UnadjRatio', 'GARTYPE', 'FINBSMNT', 'BASEMENT', 'LNDSIZE']
        for col in required_columns:
            if col not in df_sales.columns:
                df_sales[col] = np.nan

        # Step 17: Debugging output
        print("df_sales head after processing:")
        print(df_sales[['PARCEL', 'USETYPE', 'EFFBLT', 'ARCSTYLE', 'CONDITION', 'QUALITY', 'LIVEAREA', 'RATIO', 'UnadjRatio', 'GARTYPE', 'FINBSMNT', 'BASEMENT', 'LNDSIZE']].head())
        print(f"Rows in df_sales: {len(df_sales)}")
        print(f"Residential rows with EFFBLT: {df_sales[df_sales['USETYPE'] == 'RESIDENTIAL']['EFFBLT'].notna().sum()}")
        print(f"Residential rows with ARCSTYLE: {df_sales[df_sales['USETYPE'] == 'RESIDENTIAL']['ARCSTYLE'].notna().sum()}")
        print(f"Residential rows with GARTYPE: {df_sales[df_sales['USETYPE'] == 'RESIDENTIAL']['GARTYPE'].notna().sum()}")
        print(f"Rows with LNDSIZE: {df_sales['LNDSIZE'].notna().sum()}")

        return df_sales, df_current_value, bldgnum_1_array
    except Exception as e:
        print(f"Error processing data: {e}")
        return pd.DataFrame(), pd.DataFrame(), np.array([])

def compute_regression_analysis(df_sales):
    try:
        # Step 1: Compute Median Ratio for the entire dataset
        median_ratio = df_sales['RATIO'].median()  

        # Step 2: Compute requested variables
        df_sales['CURRTOT_MedianRatio'] = df_sales['CURRTOT'] / median_ratio
        df_sales['Hybrid_Value'] = 0.5 * df_sales['TASP'] + 0.5 * df_sales['CURRTOT_MedianRatio']
        
        # Independent variable: ln(value) / 0.693
        df_sales['LN_Value'] = np.where(
            df_sales['Hybrid_Value'] > 0,
            np.log(df_sales['Hybrid_Value']) / 0.693,
            np.nan
        )
        
        # Dependent variable: (Ratio - MedianRatio) / MedianRatio
        df_sales['Dependent_Var'] = (df_sales['RATIO'] - median_ratio) / median_ratio

        # Step 3: Prepare data for regression (drop NaN values)
        regression_data = df_sales[['LN_Value', 'Dependent_Var', 'Hybrid_Value', 'RATIO']].dropna()

        if regression_data.empty:
            print("Error: No valid data available for regression analysis after dropping NaN values.")
            return pd.DataFrame(columns=['Model', 'Unstandardized Coefficients (B)', 'Unstandardized Coefficients (Std. Error)',
                                        'Standardized Coefficients (Beta)', 't-value', 'Sig.']), {}, {}

        # Step 4: Perform regression analysis
        X = sm.add_constant(regression_data['LN_Value'])  # Add constant for intercept
        y = regression_data['Dependent_Var']
        model = sm.OLS(y, X).fit()

        # Step 5: Extract regression results for table
        results = {
            'Model': ['LN_Value'],
            'Unstandardized Coefficients (B)': [model.params['LN_Value']],
            'Unstandardized Coefficients (Std. Error)': [model.bse['LN_Value']],
            'Standardized Coefficients (Beta)': [model.params['LN_Value'] * regression_data['LN_Value'].std() / regression_data['Dependent_Var'].std()],
            't-value': [model.tvalues['LN_Value']],
            'Sig.': [model.pvalues['LN_Value']]
        }

        # Step 6: Format results into a DataFrame
        results_df = pd.DataFrame(results)
        results_df['Unstandardized Coefficients (B)'] = results_df['Unstandardized Coefficients (B)'].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        results_df['Unstandardized Coefficients (Std. Error)'] = results_df['Unstandardized Coefficients (Std. Error)'].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        results_df['Standardized Coefficients (Beta)'] = results_df['Standardized Coefficients (Beta)'].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        results_df['t-value'] = results_df['t-value'].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
        results_df['Sig.'] = results_df['Sig.'].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

        # Step 7: Generate model summary
        model_summary = {
            'R': np.sqrt(model.rsquared),
            'R-squared': model.rsquared,
            'Adjusted R-squared': model.rsquared_adj,
            'Std. Error of the Estimate': np.sqrt(model.mse_resid),
            'F-statistic': model.fvalue,
            'Sig. (F-statistic)': model.f_pvalue,
            'N': len(regression_data)
        }
        model_summary = {k: f"{v:.4f}" if isinstance(v, (int, float)) else v for k, v in model_summary.items()}

        # Step 8: Generate scatter plots
        os.makedirs("static", exist_ok=True)  # Ensure static folder exists

        # Scatter Plot 1: Hybrid_Value (x) vs Ratio (y)
        plt.figure(figsize=(6, 4))
        plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
        plt.scatter(regression_data['Hybrid_Value'], regression_data['RATIO'], alpha=0.5, color='black', marker='x', s=20)
        plt.xlabel('Hybrid Value = .5(TASP) + .5(CURRTOT/Median)') 
        plt.ylabel('Ratio')
        plt.ylim(0, 3)
        plt.title('Hybrid Value vs Ratio')
        hybrid_plot_path = os.path.join("static", "hybrid_value_vs_ratio.png")
        plt.savefig(hybrid_plot_path)
        plt.close()

        # Scatter Plot 2: LN_Value (x) vs Dependent_Var (y) with regression line
        slope = model.params['LN_Value']
        intercept = model.params['const']
        x_range = np.array([regression_data['LN_Value'].min(), regression_data['LN_Value'].max()])
        y_pred = slope * x_range + intercept

        plt.figure(figsize=(6, 4))
        plt.scatter(regression_data['LN_Value'], regression_data['Dependent_Var'], alpha=0.5, color='orange', marker='x', s=20)
        plt.plot(x_range, y_pred, color='black', linestyle='--', 
                 label=f'Regression (y = {slope:.4f}x + {intercept:.4f})')
        plt.axhline(y=0, color='red', linestyle='-', linewidth=1)  # Changed to y=0 since dependent var is (Ratio-Median)/Median
        plt.xlabel('LN("Value")/.693')
        plt.ylabel('(Ratio-Median)/Median')
        plt.title('LN_Value vs Dependent Variable')
        plt.legend()
        ln_value_plot_path = os.path.join("static", "ln_value_vs_dependent_var.png")
        plt.savefig(ln_value_plot_path)
        plt.close()

        # Step 9: Return results and plot paths
        plot_paths_coeffofprb = {
            'hybrid_value_vs_ratio': f"/static/hybrid_value_vs_ratio.png",
            'ln_value_vs_dependent_var': f"/static/ln_value_vs_dependent_var.png"
        }

        return results_df, model_summary, plot_paths_coeffofprb

    except Exception as e:
        print(f"Error in regression analysis: {e}")
        return pd.DataFrame(columns=['Model', 'Unstandardized Coefficients (B)', 'Unstandardized Coefficients (Std. Error)',
                                    'Standardized Coefficients (Beta)', 't-value', 'Sig.']), {}, {}

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
        'Mean': round(group['PCTCHANGE'].mean(), 4),
        'Median': round(group['PCTCHANGE'].median(), 4),
        'Min': round(group['PCTCHANGE'].min(), 4),
        'Max': round(group['PCTCHANGE'].max(), 4)
    }

def compute_pctchange_stats_nonsold(group):
    return {
        'N': len(group),
        'Mean': round(group['PCTCHANGE'].mean(), 4),
        'Median': round(group['PCTCHANGE'].median(), 4),
        'Min': round(group['PCTCHANGE'].min(), 4),
        'Max': round(group['PCTCHANGE'].max(), 4)
    }
def compute_arcstyle_stats(group):
    return {
        'ARCSTYLE': group['ARCSTYLE'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

# Helper function to assign EFFBLT ranges
def assign_effblt_range(effblt):
    """Convert an EFFBLT year into a range like 0-9, 10-19, etc., or '60 and up'."""
    if pd.isna(effblt):
        return "Unknown"
    
    effblt = int(effblt)  # Ensure it's an integer
    
    if effblt < 60:
        lower_bound = (effblt // 10) * 10
        upper_bound = lower_bound + 9
        return f"{lower_bound}-{upper_bound}"
    else:
        return "60 and up"
    
def assign_val_range(value):
    """Convert an EFFBLT year into a range like 0-9, 10-19, etc., or '60 and up'."""
    if pd.isna(value):
        return "Unknown"
    
    value = int(value)  # Ensure it's an integer
    
    if value < 2000:
        lower_bound = (value // 500) * 500
        upper_bound = lower_bound + 499
        return f"{lower_bound}-{upper_bound}"
    else:
        return "2000 and up"
    
def assign_lnd_range(value):
    """Convert an EFFBLT year into a range like 0-9, 10-19, etc., or '60 and up'."""
    if pd.isna(value):
        return "Unknown"
    
    value = int(value)  # Ensure it's an integer
    
    if value < 10000:
        lower_bound = (value // 2500) * 2500
        upper_bound = lower_bound + 2499
        return f"{lower_bound}-{upper_bound}"
    else:
        return "10000 and up"


# EFFBLT stats computation
def compute_effblt_stats(group):
    """Compute statistics for a group of data based on EFFBLT ranges."""
    return {
        'EFFBLT': group['EFFBLT_RANGE'].iloc[0],  # Use the range instead of raw EFFBLT
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def compute_quality_stats(group):
    return {
        'QUALITY': group['QUALITY'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def compute_condition_stats(group):
    return {
        'CONDITION': group['CONDITION'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def compute_gartype_stats(group):
    return {
        'GARTYPE': group['GARTYPE'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }


def compute_finbsmnt_stats(group):
    return {
        'FINBSMNT': group['FINBSMNT_RANGE'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def compute_bsmnt_stats(group):
    return {
        'BASEMENT': group['BASEMENT_RANGE'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def compute_livarea_stats(group):
    return {
        'LIVEAREA': group['LIVEAREA_RANGE'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def compute_lndsize_stats(group):
    return {
        'LNDSIZE': group['LNDSIZE_Range'].iloc[0],
        'N': len(group),
        'Mean': round(group['RATIO'].mean(), 2),
        'Median': round(group['RATIO'].median(), 2),
        'Min': round(group['RATIO'].min(), 2),
        'Max': round(group['RATIO'].max(), 2),
    }

def calculate_quality_totals(data):
    if not data:
        return {'QUALITY': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'QUALITY': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_condition_totals(data):
    if not data:
        return {'CONDITION': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'CONDITION': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_arcstyle_totals(data):
    if not data:
        return {'ARCSTYLE': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'ARCSTYLE': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_gartype_totals(data):
    if not data:
        return {'GARTYPE': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'GARTYPE': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_finbsmnt_totals(data):
    if not data:
        return {'FINBSMNT': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'FINBSMNT': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_finbsmnt_totals(data):
    if not data:
        return {'FINBSMNT': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'FINBSMNT': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_bsmnt_totals(data):
    if not data:
        return {'BASEMENT': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'BASEMENT': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_livearea_totals(data):
    if not data:
        return {'LIVEAREA': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'LIVEAREA': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

def calculate_lndsize_totals(data):
    if not data:
        return {'LNDSIZE': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'LNDSIZE': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
    }

# EFFBLT totals calculation
def calculate_effblt_totals(data):
    """Calculate totals for EFFBLT range data."""
    if not data:
        return {'EFFBLT': 'Total', 'N': 0, 'Mean': 0, 'Median': 0, 'Min': 0, 'Max': 0}
    total_n = sum(row['N'] for row in data)
    total_mean = sum(row['Mean'] for row in data) / len(data)
    total_median = sum(row['Median'] for row in data) / len(data)
    total_min = min(row['Min'] for row in data)
    total_max = max(row['Max'] for row in data)
    return {
        'EFFBLT': 'Total',
        'N': total_n,
        'Mean': round(total_mean, 2),
        'Median': round(total_median, 2),
        'Min': round(total_min, 2),
        'Max': round(total_max, 2),
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


def generate_pdf_report(file_id, month_filters, county_name):
    # Define page margins to match footer line (left=50, right=50, bottom=60 to avoid footer overlap)
    pdf_path = os.path.join("static", f"report_{file_id}.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=36,
        bottomMargin=60  # Ensure content doesn't overlap with footer
    )
    elements = []
    styles = getSampleStyleSheet()

    TOTAL_PAGES = 15  # Updated to account for regression analysis pages

    # Define the footer drawing function
    def add_footer(canvas, doc):
        canvas.saveState()
        
        # Draw a horizontal line at the bottom
        canvas.setStrokeColor(colors.grey)
        canvas.setLineWidth(0.5)
        canvas.line(50, 50, 550, 50)  # Line from 50 to 550 points

        # Set font and size
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.black)
        
        # Add current date on the left side
        current_date = datetime.now().strftime("%m-%d-%Y")  # Get the current date in YYYY-MM-DD format
        canvas.drawString(50, 30, current_date)  # Left-aligned at x=50

        canvas.drawCentredString(300, 30, county_name)  # Centered at x=300

        # Add page number on the right side (just the number)
        page_number = f"Page {doc.page} of {TOTAL_PAGES}"
        canvas.drawRightString(550, 30, page_number)  # Right-aligned at x=550
        
        canvas.restoreState()

    # Load data
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    if not os.path.exists(file_path):
        return "Data file not found."
    df_sales = pd.read_json(file_path, orient='split')

    current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"current_{file_id}.json")
    if not os.path.exists(current_file_path):
        return "Current value file not found."
    df_current_value = pd.read_json(current_file_path, orient='split')

    # Compute tables
    usetypes = df_sales['USETYPE'].unique()
    table1_data = {usetype: [compute_statistics(group) for _, group in df_sales[df_sales['USETYPE'] == usetype].groupby('SYEAR')] for usetype in usetypes}

    table1_totals = {}
    for usetype in usetypes:
        rows = table1_data[usetype]
        total_n = sum(row['N'] for row in rows)
        total_median_ratio = sum(row['Median Ratio'] for row in rows) / len(rows) if rows else 0
        total_cod = sum(row['COD'] for row in rows) / len(rows) if rows else 0
        total_mean_ratio = sum(row['Mean Ratio'] for row in rows) / len(rows) if rows else 0
        total_weighted_mean_ratio = sum(row['Weighted Mean Ratio'] for row in rows) / len(rows) if rows else 0
        total_prd = sum(row['PRD'] for row in rows) / len(rows) if rows else 0
        table1_totals[usetype] = {
            'SYEAR': 'Total',
            'N': total_n,
            'Median Ratio': round(total_median_ratio, 2),
            'COD': round(total_cod, 2),
            'Mean Ratio': round(total_mean_ratio, 2),
            'Weighted Mean Ratio': round(total_weighted_mean_ratio, 2),
            'PRD': round(total_prd, 2)
        }

    table2_data = {}
    for usetype in usetypes:
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        total_n = len(filtered_data)
        total_mean = filtered_data['PCTCHANGE'].mean()
        total_median = filtered_data['PCTCHANGE'].median()
        total_min = filtered_data['PCTCHANGE'].min()
        total_max = filtered_data['PCTCHANGE'].max()
        table2_data[usetype] = {
            'USETYPE': usetype,
            'N': total_n,
            'Mean': round(total_mean, 4),
            'Median': round(total_median, 4),
            'Min': round(total_min, 4),
            'Max': round(total_max, 4)
        }

    # Non-sold properties
    sold_parcels = df_sales['PARCEL'].unique()
    df_non_sold = df_current_value[~df_current_value['PARCEL'].isin(sold_parcels)].copy()
    df_non_sold['USETYPE'] = df_non_sold['ABSTRPRD'].apply(assign_usetype)
    df_non_sold = df_non_sold[df_non_sold['USETYPE'].isin(['RESIDENTIAL', 'COMMERCIAL', 'VACANT LAND'])]
    summary_data = df_non_sold.groupby('USETYPE')['PCTCHANGE'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index().round(4).to_dict('records')

    combined_data = {}
    for usetype in usetypes:
        combined_data[usetype] = []
        sold_row = table2_data[usetype].copy()
        sold_row['Type'] = 'Sold'
        combined_data[usetype].append(sold_row)
        for row in summary_data:
            if row['USETYPE'] == usetype:
                unsold_row = row.copy()
                unsold_row['Type'] = 'Unsold'
                combined_data[usetype].append(unsold_row)

    # ARCSTYLE and EFFBLT for RESIDENTIAL
    residential_sales = df_sales[df_sales['USETYPE'] == 'RESIDENTIAL'].copy()
    residential_sales['EFFBLT_RANGE'] = residential_sales['EFFBLT'].apply(assign_effblt_range)
    residential_sales['FINBSMNT_RANGE'] = residential_sales['FINBSMNT'].apply(assign_val_range)
    residential_sales['BASEMENT_RANGE'] = residential_sales['BASEMENT'].apply(assign_val_range)
    residential_sales['LIVEAREA_RANGE'] = residential_sales['LIVEAREA'].apply(assign_val_range)
    residential_sales['LNDSIZE_Range'] = residential_sales['LNDSIZE'].apply(assign_lnd_range)
    
    arcstyle_data = [compute_arcstyle_stats(group) for _, group in residential_sales.groupby('ARCSTYLE') if pd.notna(group['ARCSTYLE'].iloc[0])]
    effblt_data = [compute_effblt_stats(group) for _, group in residential_sales.groupby('EFFBLT_RANGE') if group['EFFBLT_RANGE'].iloc[0] != "Unknown"]
    quality_data = [compute_quality_stats(group) for _, group in residential_sales.groupby('QUALITY') if pd.notna(group['QUALITY'].iloc[0])]
    condition_data = [compute_condition_stats(group) for _, group in residential_sales.groupby('CONDITION') if pd.notna(group['CONDITION'].iloc[0])]
    gartype_data = [compute_gartype_stats(group) for _, group in residential_sales.groupby('GARTYPE') if pd.notna(group['GARTYPE'].iloc[0])]
    finbsmnt_data = [compute_finbsmnt_stats(group) for _, group in residential_sales.groupby('FINBSMNT_RANGE') if pd.notna(group['FINBSMNT_RANGE'].iloc[0])]
    bsmnt_data = [compute_bsmnt_stats(group) for _, group in residential_sales.groupby('BASEMENT_RANGE') if pd.notna(group['BASEMENT_RANGE'].iloc[0])]
    lndsize_data = [compute_lndsize_stats(group) for _, group in residential_sales.groupby('LNDSIZE_Range') if pd.notna(group['LNDSIZE_Range'].iloc[0])]
    livearea_data = [compute_livarea_stats(group) for _, group in residential_sales.groupby('LIVEAREA_RANGE') if pd.notna(group['LIVEAREA_RANGE'].iloc[0])]
    arcstyle_totals = calculate_arcstyle_totals(arcstyle_data)
    effblt_totals = calculate_effblt_totals(effblt_data)
    quality_totals = calculate_quality_totals(quality_data)
    condition_totals = calculate_condition_totals(condition_data)
    gartype_totals = calculate_gartype_totals(gartype_data)
    finbsmnt_totals = calculate_finbsmnt_totals(finbsmnt_data)
    bsmnt_totals = calculate_bsmnt_totals(bsmnt_data)
    livearea_totals = calculate_livearea_totals(livearea_data)
    lndsize_totals = calculate_lndsize_totals(lndsize_data)


    # Scatter plots
    scatter_plots = {}
    usetypes = df_sales['USETYPE'].unique()
    colors_map = {'RESIDENTIAL': 'green', 'COMMERCIAL': 'red', 'VACANT LAND': 'blue'}

    for usetype in usetypes:
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        max_months = filtered_data['MONTHS'].max()
        
        slope = np.nan
        intercept = np.nan
        if 'MONTHS' in filtered_data.columns and 'RATIO' in filtered_data.columns:
            regression_data = filtered_data[['MONTHS', 'RATIO']].dropna()
            
            if not regression_data.empty:
                X = regression_data['MONTHS'].values
                y = regression_data['RATIO'].values

                X_reshaped = X.reshape(-1, 1)
                model = LinearRegression()
                model.fit(X_reshaped, y)

                slope = model.coef_[0]
                intercept = model.intercept_
                r_squared = model.score(X_reshaped, y)
                correlation_coefficient = np.corrcoef(X, y)[0, 1]

        plot_path = os.path.join("static", f"{usetype}_scatter.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(filtered_data['MONTHS'], filtered_data['RATIO'], 
                    alpha=0.5, 
                    color=colors_map.get(usetype, 'black'), 
                    marker='x', 
                    s=20) 
        
        if not np.isnan(slope):
            x_range = np.array([0, max_months])
            y_pred = slope * x_range + intercept
            plt.plot(x_range, y_pred, color='black', linestyle='--', 
                    label=f'Regression (y = {slope:.4f}x + {intercept:.4f})\nR² = {r_squared:.6f}\nCorrelation Coefficient = {correlation_coefficient:.4f}')
        
        plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
        plt.xlabel('Months')
        plt.ylabel('Ratio')
        plt.ylim(0, 3)
        plt.xlim(max_months, 0)
        plt.title(f'Months vs Ratio')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
        scatter_plots[usetype] = plot_path

    scatter_plots_unadjratio = {}
    for usetype in usetypes:
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        max_months = filtered_data['MONTHS'].max()

        slope_u = np.nan
        intercept_u = np.nan
        r_squared_u = np.nan
        correlation_coefficient_u = np.nan
        
        if 'MONTHS' in filtered_data.columns and 'UnadjRatio' in filtered_data.columns:
            regression_data_u = filtered_data[['MONTHS', 'UnadjRatio']].dropna()
            
            if not regression_data_u.empty:
                X = regression_data_u['MONTHS'].values.reshape(-1, 1)
                y = regression_data_u['UnadjRatio'].values

                model = LinearRegression()
                model.fit(X, y)

                slope_u = model.coef_[0]
                intercept_u = model.intercept_
                r_squared_u = model.score(X, y)
                correlation_coefficient_u = np.corrcoef(regression_data_u['MONTHS'], y)[0, 1]
        
        plot_path_u = os.path.join("static", f"{usetype}_scatter_unadjratio.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(filtered_data['MONTHS'], filtered_data['UnadjRatio'], alpha=0.5, 
                    color=colors_map.get(usetype, 'black'), marker='x', s=20)
        
        if not np.isnan(slope_u):
            x_range_u = np.array([0, max_months])
            y_pred_u = slope_u * x_range_u + intercept_u
            plt.plot(x_range_u, y_pred_u, color='black', linestyle='--', 
                     label=f'Regression (y = {slope_u:.4f}x + {intercept_u:.4f})\nR² = {r_squared_u:.6f}\nCorrelation Coeff = {correlation_coefficient_u:.4f}')
        
        plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
        plt.xlabel('Months')
        plt.ylabel('Unadjusted Ratio')
        plt.ylim(0, 3)
        plt.xlim(max_months, 0)
        plt.title(f'Months vs Unadjusted Ratio')
        plt.legend()
        plt.savefig(plot_path_u)
        plt.close()
        scatter_plots_unadjratio[usetype] = plot_path_u    

    # Filter data for RESIDENTIAL scatter plots
    filtered_data1 = df_sales[df_sales['USETYPE'] == 'RESIDENTIAL']

    # EFFBLT scatter
    plot_path1 = os.path.join("static", "effblt_scatter.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['EFFBLT'], filtered_data1['RATIO'], alpha=0.5, marker='x', s=20, c='orange')
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Effblt')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Effblt')
    plt.legend()
    plt.savefig(plot_path1)
    plt.close()
    scatter_plot_effblt = plot_path1

    # Live Area scatter
    plot_path2 = os.path.join("static", "livearea_scatter.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['LIVEAREA'], filtered_data1['RATIO'], alpha=0.5, marker='x', s=20, c='purple')
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Living Area')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Living Area')
    plt.legend()
    plt.savefig(plot_path2)
    plt.close()
    scatter_plot_livearea = plot_path2

    # Define plot path for FINBSMNT scatter
    plot_path3 = os.path.join("static", "finbsmnt_scatter.png")

    # Create LFINBSMNT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['FINBSMNT'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='yellow')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Finbsmnt')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Finbsmnt')
    plt.legend()
    plt.savefig(plot_path3)
    plt.close()
    scatter_plot_finbsmnt = plot_path3

    plot_path4 = os.path.join("static", "bsmnt_scatter.png")

    # Create LFINBSMNT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['BASEMENT'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='violet')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Basement')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for bsmnt')
    plt.legend()
    plt.savefig(plot_path4)
    plt.close()
    scatter_plot_bsmnt = plot_path4

    plot_path5 = os.path.join("static", "lndsize_scatter.png")

    # Create LFINBSMNT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['LNDSIZE'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='brown')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Land Size')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Land Size')
    plt.legend()
    plt.savefig(plot_path5)
    plt.close()
    scatter_plot_lndsize = plot_path4

    # Perform regression analysis
    regression_results, model_summary,regression_plots = compute_regression_analysis(df_sales)

    # PDF content
    elements.append(Paragraph("Audit Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    for usetype in usetypes:
        elements.append(Paragraph(f"Ratios Table for {usetype}", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        data = [["SYEAR", "N", "Mean Ratio","Median Ratio", "Weighted Mean Ratio","COD","PRD"]]
        for row in table1_data[usetype]:
            data.append([row['SYEAR'], row['N'], row['Mean Ratio'], row['Median Ratio'], 
                         row['Weighted Mean Ratio'], row['COD'], row['PRD']])
        data.append([table1_totals[usetype]['SYEAR'], table1_totals[usetype]['N'], 
                     table1_totals[usetype]['Mean Ratio'], table1_totals[usetype]['Median Ratio'], 
                     table1_totals[usetype]['Weighted Mean Ratio'], table1_totals[usetype]['COD'], 
                     table1_totals[usetype]['PRD']])
        table = Table(data, colWidths=[50, 50, 80, 80, 120, 40, 50])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Percentage Change and Market Value for {usetype}", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        data2 = [["Type", "N", "Mean", "Median", "Min", "Max"]]
        for row in combined_data[usetype]:
            data2.append([row['Type'], row['N'] if 'N' in row else row['count'], 
                          row['Mean'] if 'Mean' in row else row['mean'], 
                          row['Median'] if 'Median' in row else row['median'], 
                          row['Min'] if 'Min' in row else row['min'], 
                          row['Max'] if 'Max' in row else row['max']])
        table2 = Table(data2, colWidths=[80, 60, 80, 80, 80, 80])
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
        ]))
        elements.append(table2)
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Scatter Plot for {usetype}", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        if os.path.exists(scatter_plots[usetype]):
            elements.append(Image(scatter_plots[usetype], width=400, height=300))
        
        elements.append(Spacer(1, 48))

        elements.append(Paragraph(f"Scatter plot of months (x) vs UnadjRatio for {usetype}", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        if os.path.exists(scatter_plots_unadjratio[usetype]):
            elements.append(Image(scatter_plots_unadjratio[usetype], width=400, height=300))

        elements.append(PageBreak())

        if usetype == 'RESIDENTIAL':
            elements.append(Paragraph("Grouping Data for Residential based on ARCSTYLE", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            arc_data = [["ARCSTYLE", "N", "Mean", "Median", "Min", "Max"]]
            if arcstyle_data:
                for row in arcstyle_data:
                    arc_data.append([row['ARCSTYLE'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if arcstyle_totals:
                    arc_data.append([arcstyle_totals['ARCSTYLE'], arcstyle_totals['N'], 
                                     arcstyle_totals['Mean'], arcstyle_totals['Median'], 
                                     arcstyle_totals['Min'], arcstyle_totals['Max']])
            else:
                arc_data.append(["No Data", 0, 0, 0, 0, 0])
            arc_table = Table(arc_data, colWidths=[120, 60, 80, 80, 80, 80])
            arc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(arc_table)
            elements.append(Spacer(1, 12))
            elements.append(PageBreak())

            elements.append(Paragraph("Grouping Data for Residential based on EFFBLT", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            eff_data = [["EFFBLT Range", "N", "Mean", "Median", "Min", "Max"]]
            if effblt_data:
                for row in effblt_data:
                    eff_data.append([row['EFFBLT'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if effblt_totals:
                    eff_data.append([effblt_totals['EFFBLT'], effblt_totals['N'], 
                                     effblt_totals['Mean'], effblt_totals['Median'], 
                                     effblt_totals['Min'], effblt_totals['Max']])
            else:
                eff_data.append(["No Data", 0, 0, 0, 0, 0])
            eff_table = Table(eff_data, colWidths=[100, 60, 80, 80, 80, 80])
            eff_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(eff_table)
            elements.append(Spacer(1, 16))

            elements.append(Paragraph("Grouping Data for Residential based on CONDITION", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            con_data = [["CONDITION", "N", "Mean", "Median", "Min", "Max"]]
            if condition_data:
                for row in condition_data:
                    con_data.append([row['CONDITION'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if condition_totals:
                    con_data.append([condition_totals['CONDITION'], condition_totals['N'], 
                                     condition_totals['Mean'], condition_totals['Median'], 
                                     condition_totals['Min'], condition_totals['Max']])
            else:
                con_data.append(["No Data", 0, 0, 0, 0, 0])
            con_table = Table(con_data, colWidths=[120, 60, 80, 80, 80, 80])
            con_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(con_table)
            elements.append(Spacer(1, 16))

            elements.append(Paragraph("Grouping Data for Residential based on QUALITY", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            qual_data = [["QUALITY", "N", "Mean", "Median", "Min", "Max"]]
            if quality_data:
                for row in quality_data:
                    qual_data.append([row['QUALITY'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if quality_totals:
                    qual_data.append([quality_totals['QUALITY'], quality_totals['N'], 
                                     quality_totals['Mean'], quality_totals['Median'], 
                                     quality_totals['Min'], quality_totals['Max']])
            else:
                qual_data.append(["No Data", 0, 0, 0, 0, 0])
            qual_table = Table(qual_data, colWidths=[120, 60, 80, 80, 80, 80])
            qual_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(qual_table)
            elements.append(Spacer(1, 12))

            elements.append(PageBreak())

            elements.append(Paragraph("Grouping Data for Residential based on GARTYPE", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            gt_data = [["GARTYPE", "N", "Mean", "Median", "Min", "Max"]]
            if gartype_data:
                for row in gartype_data:
                    gt_data.append([row['GARTYPE'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if gartype_totals:
                    gt_data.append([gartype_totals['GARTYPE'], gartype_totals['N'], 
                                     gartype_totals['Mean'], gartype_totals['Median'], 
                                     gartype_totals['Min'], gartype_totals['Max']])
            else:
                gt_data.append(["No Data", 0, 0, 0, 0, 0])
            gt_table = Table(gt_data, colWidths=[120, 60, 80, 80, 80, 80])
            gt_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(gt_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Grouping Data for Residential based on FIN BASEMENT", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            fb_data = [["FINBSMNT", "N", "Mean", "Median", "Min", "Max"]]
            if finbsmnt_data:
                for row in finbsmnt_data:
                    fb_data.append([row['FINBSMNT'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if finbsmnt_totals:
                    fb_data.append([finbsmnt_totals['FINBSMNT'], finbsmnt_totals['N'], 
                                     finbsmnt_totals['Mean'], finbsmnt_totals['Median'], 
                                     finbsmnt_totals['Min'], finbsmnt_totals['Max']])
            else:
                fb_data.append(["No Data", 0, 0, 0, 0, 0])
            fb_table = Table(fb_data, colWidths=[120, 60, 80, 80, 80, 80])
            fb_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(fb_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Grouping Data for Residential based on BASEMENT", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            b_data = [["BASEMENT", "N", "Mean", "Median", "Min", "Max"]]
            if bsmnt_data:
                for row in bsmnt_data:
                    b_data.append([row['BASEMENT'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if bsmnt_totals:
                    b_data.append([bsmnt_totals['BASEMENT'], bsmnt_totals['N'], 
                                     bsmnt_totals['Mean'], bsmnt_totals['Median'], 
                                     bsmnt_totals['Min'], bsmnt_totals['Max']])
            else:
                b_data.append(["No Data", 0, 0, 0, 0, 0])
            b_table = Table(b_data, colWidths=[120, 60, 80, 80, 80, 80])
            b_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(b_table)
            elements.append(Spacer(1, 12))

            elements.append(PageBreak())

            elements.append(Paragraph("Grouping Data for Residential based on LIVEAREA", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            la_data = [["LIVEAREA", "N", "Mean", "Median", "Min", "Max"]]
            if livearea_data:
                for row in livearea_data:
                    la_data.append([row['LIVEAREA'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if livearea_totals:
                    la_data.append([livearea_totals['LIVEAREA'], livearea_totals['N'], 
                                    livearea_totals['Mean'], livearea_totals['Median'], 
                                    livearea_totals['Min'], livearea_totals['Max']])
            else:
                la_data.append(["No Data", 0, 0, 0, 0, 0])
            la_table = Table(la_data, colWidths=[120, 60, 80, 80, 80, 80])
            la_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(la_table)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Grouping Data for Residential based on LAND SIZE", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            ls_data = [["LNDSIZE", "N", "Mean", "Median", "Min", "Max"]]
            if lndsize_data:
                for row in lndsize_data:
                    ls_data.append([row['LNDSIZE'], row['N'], row['Mean'], row['Median'], row['Min'], row['Max']])
                if lndsize_totals:
                    ls_data.append([lndsize_totals['LNDSIZE'], lndsize_totals['N'], 
                                    lndsize_totals['Mean'], lndsize_totals['Median'], 
                                    lndsize_totals['Min'], lndsize_totals['Max']])
            else:
                ls_data.append(["No Data", 0, 0, 0, 0, 0])
            ls_table = Table(ls_data, colWidths=[120, 60, 80, 80, 80, 80])
            ls_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -2), colors.HexColor("#f7fafc")),
                ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#cbd5e0")),
            ]))
            elements.append(ls_table)
            elements.append(Spacer(1, 12))


            elements.append(PageBreak())

            # Add scatter plots
            elements.append(Paragraph("Scatter Plot for Effective Built Year (EFFBLT)", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            elements.append(Image("static/effblt_scatter.png", width=300, height=250))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Scatter Plot for Living Area", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            elements.append(Image("static/livearea_scatter.png", width=300, height=250))
            elements.append(Spacer(1, 12))

            elements.append(PageBreak())

            elements.append(Paragraph("Scatter Plot for Fin Basement", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            elements.append(Image("static/finbsmnt_scatter.png", width=300, height=250))
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Scatter Plot for Basement", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            elements.append(Image("static/bsmnt_scatter.png", width=300, height=250))
            elements.append(Spacer(1, 12))

            elements.append(PageBreak())

            elements.append(Paragraph("Scatter Plot for Land Size", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            elements.append(Image("static/lndsize_scatter.png", width=300, height=250))
            elements.append(Spacer(1, 12))
            
            elements.append(PageBreak())
        
    # Add regression analysis section
    elements.append(Paragraph("PRB Regression Analysis", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Add Model Summary section
    elements.append(Paragraph("Model Summary", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    model_summary_data = [
        ["R", "R-squared", "Adjusted R-squared", "Std. Error"],
        [
            model_summary.get('R', 'N/A'),
            model_summary.get('R-squared', 'N/A'),
            model_summary.get('Adjusted R-squared', 'N/A'),
            model_summary.get('Std. Error of the Estimate', 'N/A'),
        ]
    ]

    summary_table = Table(model_summary_data, colWidths=[130, 130, 140, 130])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#f7fafc")),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 12))

    # Add Coefficients section
    elements.append(Paragraph("Coefficients", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    regression_data = [
        ["Model", "Unstandardized Coefficients", "", "Standardized Coefficients", "t-value", "Sig."],
        ["", "B", "Std. Error", "Beta", "", ""]
    ]

    if not regression_results.empty:
        for _, row in regression_results.iterrows():
            regression_data.append([
                row['Model'],
                row['Unstandardized Coefficients (B)'] if isinstance(row['Unstandardized Coefficients (B)'], str) 
                    else f"{row['Unstandardized Coefficients (B)']:.4f}",
                row['Unstandardized Coefficients (Std. Error)'] if isinstance(row['Unstandardized Coefficients (Std. Error)'], str) 
                    else f"{row['Unstandardized Coefficients (Std. Error)']:.4f}",
                row['Standardized Coefficients (Beta)'] if isinstance(row['Standardized Coefficients (Beta)'], str) 
                    else f"{row['Standardized Coefficients (Beta)']:.4f}",
                row['t-value'] if isinstance(row['t-value'], str) else f"{row['t-value']:.4f}",
                row['Sig.'] if isinstance(row['Sig.'], str) else f"{row['Sig.']:.4f}"
            ])
    else:
        regression_data.append(["No regression results available", "", "", "", "", ""])

    reg_table = Table(regression_data, colWidths=[80, 100, 100, 150, 50, 50])

    reg_table.setStyle(TableStyle([
        # Header row 1
        ('SPAN', (1, 0), (2, 0)),  # Span Unstandardized Coefficients
        ('SPAN', (3, 0), (3, 0)),  # Span Standardized Coefficients
        ('SPAN', (0, 0), (0, 0)),  # Span Model
        ('SPAN', (4, 0), (4, 0)),  # Span t-value
        ('SPAN', (5, 0), (5, 0)),  # Span Sig.

        # Style headers
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4a5568")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),

        # Table grid and box
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),

        # Background colors
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#e2e8f0")),  # subheader row
        ('BACKGROUND', (0, 2), (-1, -1), colors.HexColor("#f7fafc")),  # data rows
    ]))

    elements.append(reg_table)
    elements.append(Spacer(1, 12))

    # Add plots
    elements.append(Paragraph("PRB Plot of Ratios against value proxy as percentages", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    if regression_plots and os.path.exists(regression_plots.get('ln_value_vs_dependent_var', '').lstrip('/')):
        elements.append(Image(regression_plots['ln_value_vs_dependent_var'].lstrip('/'), width=350, height=300))
    else:
        elements.append(Paragraph("Plot not available", styles["Normal"]))

    elements.append(PageBreak())

    elements.append(Paragraph("PRB Plot of Ratios against value proxy", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    if regression_plots and os.path.exists(regression_plots.get('hybrid_value_vs_ratio', '').lstrip('/')):
        elements.append(Image(regression_plots['hybrid_value_vs_ratio'].lstrip('/'), width=350, height=300))
    else:
        elements.append(Paragraph("Plot not available", styles["Normal"]))
    
    # Build the document with the footer
    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    return pdf_path

# Report route function
@app.route('/report/<string:file_id>')
def report(file_id):
    if not file_id:
        return "No data found. Please process the data first.", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.json")
    if not os.path.exists(file_path):
        return "Data file not found.", 404
    df_sales = pd.read_json(file_path, orient='split')

    current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"current_{file_id}.json")
    if not os.path.exists(current_file_path):
        return "Current value file not found.", 404
    df_current_value = pd.read_json(current_file_path, orient='split')

    regression_results,model_summary,plot_paths_coeffofprb = compute_regression_analysis(df_sales)

    usetypes = df_sales['USETYPE'].unique()
    table1_data = {usetype: [compute_statistics(group) for _, group in df_sales[df_sales['USETYPE'] == usetype].groupby('SYEAR')] for usetype in usetypes}

    table1_totals = {}
    for usetype in usetypes:
        rows = table1_data[usetype]
        total_n = sum(row['N'] for row in rows)
        total_median_ratio = sum(row['Median Ratio'] for row in rows) / len(rows) if rows else 0
        total_cod = sum(row['COD'] for row in rows) / len(rows) if rows else 0
        total_mean_ratio = sum(row['Mean Ratio'] for row in rows) / len(rows) if rows else 0
        total_weighted_mean_ratio = sum(row['Weighted Mean Ratio'] for row in rows) / len(rows) if rows else 0
        total_prd = sum(row['PRD'] for row in rows) / len(rows) if rows else 0
        table1_totals[usetype] = {
            'SYEAR': 'Total',
            'N': total_n,
            'Median Ratio': round(total_median_ratio, 2),
            'COD': round(total_cod, 2),
            'Mean Ratio': round(total_mean_ratio, 2),
            'Weighted Mean Ratio': round(total_weighted_mean_ratio, 2),
            'PRD': round(total_prd, 2)
        }

    table2_data = {}
    for usetype in usetypes:
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        total_n = len(filtered_data)
        total_mean = filtered_data['PCTCHANGE'].mean()
        total_median = filtered_data['PCTCHANGE'].median()
        total_min = filtered_data['PCTCHANGE'].min()
        total_max = filtered_data['PCTCHANGE'].max()
        table2_data[usetype] = {
            'USETYPE': usetype,
            'N': total_n,
            'Mean': round(total_mean, 4),
            'Median': round(total_median, 4),
            'Min': round(total_min, 4),
            'Max': round(total_max, 4)
        }

    sold_parcels = df_sales['PARCEL'].unique()
    df_non_sold = df_current_value[~df_current_value['PARCEL'].isin(sold_parcels)].copy()
    df_non_sold['USETYPE'] = df_non_sold['ABSTRPRD'].apply(assign_usetype)
    df_non_sold = df_non_sold[df_non_sold['USETYPE'].isin(['RESIDENTIAL', 'COMMERCIAL', 'VACANT LAND'])]
    summary_data = df_non_sold.groupby('USETYPE')['PCTCHANGE'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index().round(4).to_dict('records')

    combined_data = {}
    for usetype in usetypes:
        combined_data[usetype] = []
        sold_row = table2_data[usetype].copy()
        sold_row['Type'] = 'Sold'
        combined_data[usetype].append(sold_row)
        for row in summary_data:
            if row['USETYPE'] == usetype:
                unsold_row = row.copy()
                unsold_row['Type'] = 'Unsold'
                combined_data[usetype].append(unsold_row)

    residential_sales = df_sales[df_sales['USETYPE'] == 'RESIDENTIAL'].copy()
    residential_sales['EFFBLT_RANGE'] = residential_sales['EFFBLT'].apply(assign_effblt_range)
    residential_sales['FINBSMNT_RANGE'] = residential_sales['FINBSMNT'].apply(assign_val_range)
    residential_sales['BASEMENT_RANGE'] = residential_sales['BASEMENT'].apply(assign_val_range)
    residential_sales['LIVEAREA_RANGE'] = residential_sales['LIVEAREA'].apply(assign_val_range)
    residential_sales['LNDSIZE_Range'] = residential_sales['LNDSIZE'].apply(assign_lnd_range)
    
    arcstyle_data = [compute_arcstyle_stats(group) for _, group in residential_sales.groupby('ARCSTYLE') if pd.notna(group['ARCSTYLE'].iloc[0])]
    effblt_data = [compute_effblt_stats(group) for _, group in residential_sales.groupby('EFFBLT_RANGE') if group['EFFBLT_RANGE'].iloc[0] != "Unknown"]
    quality_data = [compute_quality_stats(group) for _, group in residential_sales.groupby('QUALITY') if pd.notna(group['QUALITY'].iloc[0])]
    condition_data = [compute_condition_stats(group) for _, group in residential_sales.groupby('CONDITION') if pd.notna(group['CONDITION'].iloc[0])]
    gartype_data = [compute_gartype_stats(group) for _, group in residential_sales.groupby('GARTYPE') if pd.notna(group['GARTYPE'].iloc[0])]
    finbsmnt_data = [compute_finbsmnt_stats(group) for _, group in residential_sales.groupby('FINBSMNT_RANGE') if pd.notna(group['FINBSMNT_RANGE'].iloc[0])]
    bsmnt_data = [compute_bsmnt_stats(group) for _, group in residential_sales.groupby('BASEMENT_RANGE') if pd.notna(group['BASEMENT_RANGE'].iloc[0])]
    lndsize_data = [compute_lndsize_stats(group) for _, group in residential_sales.groupby('LNDSIZE_Range') if pd.notna(group['LNDSIZE_Range'].iloc[0])]
    livearea_data = [compute_livarea_stats(group) for _, group in residential_sales.groupby('LIVEAREA_RANGE') if pd.notna(group['LIVEAREA_RANGE'].iloc[0])]
    arcstyle_totals = calculate_arcstyle_totals(arcstyle_data)
    effblt_totals = calculate_effblt_totals(effblt_data)
    quality_totals = calculate_quality_totals(quality_data)
    condition_totals = calculate_condition_totals(condition_data)
    gartype_totals = calculate_gartype_totals(gartype_data)
    finbsmnt_totals = calculate_finbsmnt_totals(finbsmnt_data) 
    bsmnt_totals = calculate_bsmnt_totals(bsmnt_data)
    livearea_totals = calculate_livearea_totals(livearea_data)
    lndsize_totals = calculate_lndsize_totals(lndsize_data)

    # Assuming this is part of your larger process_data function or a separate visualization function
    scatter_plots = {}
    usetypes = df_sales['USETYPE'].unique()  # Assuming df_sales is available from previous processing
    colors_map = {'RESIDENTIAL': 'green', 'COMMERCIAL': 'red', 'VACANT LAND': 'blue'}  # Example color mapping

    for usetype in usetypes:
        # Filter data for current usetype
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        max_months = filtered_data['MONTHS'].max()
        
        # Step 13.5: Compute slope (beta coefficient) of MONTHS vs RATIO for this usetype
        slope = np.nan  # Default value
        intercept = np.nan
        if 'MONTHS' in filtered_data.columns and 'RATIO' in filtered_data.columns:
            regression_data = filtered_data[['MONTHS', 'RATIO']].dropna()
            
            if not regression_data.empty:
                X = regression_data['MONTHS'].values  # Independent variable (convert to 1D array)
                y = regression_data['RATIO'].values   # Dependent variable

                # Reshape X for LinearRegression
                X_reshaped = X.reshape(-1, 1)

                # Fit linear regression model
                model = LinearRegression()
                model.fit(X_reshaped, y)

                # Get slope, intercept, and R-squared
                slope = model.coef_[0]
                intercept = model.intercept_
                r_squared = model.score(X_reshaped, y)

                # Compute correlation coefficient
                correlation_coefficient = np.corrcoef(X, y)[0, 1]  # Pearson correlation coefficient

                print(f"Regression Results for {usetype}:")
                print(f"Slope (m) of MONTHS vs RATIO: {slope:.4f}")
                print(f"Intercept (c): {intercept:.4f}")
                print(f"R-squared (r^2): {r_squared:.4f}")
                print(f"Correlation Coefficient (corr_coeff): {correlation_coefficient:.4f}")
            else:
                print(f"Warning: No valid data for regression for {usetype} after removing NaN values")
        else:
            print(f"Error: 'MONTHS' or 'RATIO' column missing for {usetype}")

        # Create scatter plot
        plot_path = os.path.join("static", f"{usetype}_scatter.png")
        plt.figure(figsize=(6, 4))
        
        # Scatter plot
        plt.scatter(filtered_data['MONTHS'], filtered_data['RATIO'], 
                    alpha=0.5, 
                    color=colors_map.get(usetype, 'black'), 
                    marker='x', 
                    s=20) 
        
        # Plot regression line
        if not np.isnan(slope):
            x_range = np.array([0, max_months])
            y_pred = slope * x_range + intercept
            plt.plot(x_range, y_pred, color='black', linestyle='--', 
                    label=f'Regression (y = {slope:.4f}x + {intercept:.4f})\nR² = {r_squared:.6f}\nCorrelation Coefficient = {correlation_coefficient:.4f}')
        
        # Add reference line
        plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
        
        # Customize plot
        plt.xlabel('Months')
        plt.ylabel('Ratio')
        plt.ylim(0, 3)
        plt.xlim(max_months, 0)  # Note: This reverses the x-axis
        plt.title(f'Months vs Ratio')
        plt.legend()  # Add legend to show slope and regression line
        
        # Save and close
        plt.savefig(plot_path)
        plt.close()
        scatter_plots[usetype] = f"/static/{usetype}_scatter.png"

    scatter_plots_unadjratio = {}
    for usetype in usetypes:
        filtered_data = df_sales[df_sales['USETYPE'] == usetype]
        max_months = filtered_data['MONTHS'].max()

        slope_u = np.nan  # Default value
        intercept_u = np.nan
        r_squared_u = np.nan
        correlation_coefficient_u = np.nan
        
        if 'MONTHS' in filtered_data.columns and 'UnadjRatio' in filtered_data.columns:
            regression_data_u = filtered_data[['MONTHS', 'UnadjRatio']].dropna()
            
            if not regression_data_u.empty:
                X = regression_data_u['MONTHS'].values.reshape(-1, 1)
                y = regression_data_u['UnadjRatio'].values

                model = LinearRegression()
                model.fit(X, y)

                slope_u = model.coef_[0]
                intercept_u = model.intercept_
                r_squared_u = model.score(X, y)
                correlation_coefficient_u = np.corrcoef(regression_data_u['MONTHS'], y)[0, 1]
        
        plot_path_u = os.path.join("static", f"{usetype}_scatter_unadjratio.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(filtered_data['MONTHS'], filtered_data['UnadjRatio'], alpha=0.5, 
                    color=colors_map.get(usetype, 'black'), marker='x', s=20)
        
        if not np.isnan(slope_u):
            x_range_u = np.array([0, max_months])
            y_pred_u = slope_u * x_range_u + intercept_u
            plt.plot(x_range_u, y_pred_u, color='black', linestyle='--', 
                     label=f'Regression (y = {slope_u:.4f}x + {intercept_u:.4f})\nR² = {r_squared_u:.6f}\nCorrelation Coeff = {correlation_coefficient_u:.4f}')
        
        plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
        plt.xlabel('Months')
        plt.ylabel('Unadjusted Ratio')
        plt.ylim(0, 3)
        plt.xlim(max_months, 0)
        plt.title(f'Months vs Unadjusted Ratio')
        plt.legend()  # Ensure legend appears
        
        plt.savefig(plot_path_u)
        plt.close()
        scatter_plots_unadjratio[usetype] = f"/static/{usetype}_scatter_unadjratio.png"    

    # Ensure the 'static' directory exists
    os.makedirs("static", exist_ok=True)

    # Filter data for scatter plots
    filtered_data1 = df_sales[df_sales['USETYPE'] == 'RESIDENTIAL']

    # Define plot path for EFFBLT scatter
    plot_path1 = os.path.join("static", "effblt_scatter.png")

    # Create EFFBLT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['EFFBLT'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='orange')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Effblt')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Effblt')
    plt.legend()
    plt.savefig(plot_path1)
    plt.close()

    # Define plot path for Live Area scatter
    plot_path2 = os.path.join("static", "livearea_scatter.png")

    # Create Live Area scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['LIVEAREA'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='purple')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Living Area')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Living Area')
    plt.legend()
    plt.savefig(plot_path2)
    plt.close()

    # Define plot path for FINBSMNT scatter
    plot_path3 = os.path.join("static", "finbsmnt_scatter.png")

    # Create LFINBSMNT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['FINBSMNT'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='yellow')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Finbsmnt')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Finbsmnt')
    plt.legend()
    plt.savefig(plot_path3)
    plt.close()

    # Define plot path for BSMNT scatter
    plot_path4 = os.path.join("static", "bsmnt_scatter.png")

    # Create LFINBSMNT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['BASEMENT'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='violet')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Basement')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for bsmnt')
    plt.legend()
    plt.savefig(plot_path4)
    plt.close()

    # Define plot path for LNDSIZE scatter
    plot_path5 = os.path.join("static", "lndsize_scatter.png")

    # Create LFINBSMNT scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(filtered_data1['LNDSIZE'], filtered_data1['RATIO'], 
                alpha=0.5, marker='x', s=20, 
                c='brown')  
    plt.axhline(y=1, color='red', linestyle='-', linewidth=1)
    plt.xlabel('Land Size')
    plt.ylabel('Ratio')
    plt.ylim(0, 3)
    plt.title('Scatter Plot for Land Size')
    plt.legend()
    plt.savefig(plot_path5)
    plt.close()

    scatter_plot_effblt = "/static/effblt_scatter.png"
    scatter_plot_livearea = "/static/livearea_scatter.png"
    scatter_plot_finbsmnt = "/static/finbsmnt_scatter.png"
    scatter_plot_bsmnt = "/static/bsmnt_scatter.png"
    scatter_plot_lndsize = "/static/lndsize_scatter.png"

    return render_template('report.html', 
                          usetypes=usetypes, 
                          table1_data=table1_data, 
                          table1_totals=table1_totals, 
                          combined_data=combined_data, 
                          arcstyle_data=arcstyle_data, 
                          arcstyle_totals=arcstyle_totals, 
                          effblt_data=effblt_data, 
                          gartype_data = gartype_data,
                          gartype_totals = gartype_totals,
                          finbsmnt_data = finbsmnt_data,
                          finbsmnt_totals = finbsmnt_totals,
                          bsmnt_data = bsmnt_data,
                          bsmnt_totals = bsmnt_totals,
                          lndsize_data = lndsize_data,
                          lndsize_totals = lndsize_totals,
                          livearea_data = livearea_data,
                          livearea_totals = livearea_totals,
                          effblt_totals=effblt_totals,
                          condition_data = condition_data,
                          condition_totals = condition_totals,
                          quality_data = quality_data,
                          quality_totals = quality_totals,
                          scatter_plots=scatter_plots,
                          scatter_plots_unadjratio = scatter_plots_unadjratio,
                          scatter_plot_effblt=scatter_plot_effblt, 
                          scatter_plot_livearea=scatter_plot_livearea, 
                          scatter_plot_finbsmnt = scatter_plot_finbsmnt,
                          scatter_plot_bsmnt = scatter_plot_bsmnt,
                          scatter_plot_lndsize = scatter_plot_lndsize,
                          regression_results=regression_results.to_dict(orient='records'),
                          model_summary = model_summary, 
                          plot_paths_coeffofprb = plot_paths_coeffofprb, 
                          download_link=f"/download_pdf?file_id={file_id}")

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