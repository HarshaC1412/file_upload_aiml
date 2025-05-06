from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Model for uploaded files
class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UploadedFile {self.filename}>"

# Model for storing statistical results
class StatisticalReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usetype = db.Column(db.String(50), nullable=False)
    qualified = db.Column(db.String(1), nullable=False)
    median_ratio = db.Column(db.Float, nullable=False)
    aad = db.Column(db.Float, nullable=False)
    cod = db.Column(db.Float, nullable=False)
    n_val = db.Column(db.Integer, nullable=False)
    mean_ratio = db.Column(db.Float, nullable=False)
    weighted_mean_ratio = db.Column(db.Float, nullable=False)
    std_deviation = db.Column(db.Float, nullable=False)
    variance = db.Column(db.Float, nullable=False)
    cov = db.Column(db.Float, nullable=False)
    prd = db.Column(db.Float, nullable=False)
    prb = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    syear = db.Column(db.String(4), nullable=False)

    def __repr__(self):
        return f"<StatisticalReport {self.usetype}-{self.qualified}>"

# Function to save statistical report to the database
def save_statistics_to_db(usetype, qualified, stats):
    report = StatisticalReport(
        usetype=usetype,
        qualified=qualified,
        median_ratio=stats['Median Ratio'],
        aad=stats['AAD'],
        cod=stats['COD'],
        n_val=stats['N Val'],
        mean_ratio=stats['Mean Ratio'],
        weighted_mean_ratio=stats['Weighted Mean Ratio'],
        std_deviation=stats['Standard Deviation'],
        variance=stats['Variance'],
        cov=stats['COV'],
        prd=stats['PRD'],
        prb=stats['PRB']
    )
    db.session.add(report)
    db.session.commit()
