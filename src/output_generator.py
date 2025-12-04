import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def export_all_results(df, fraud_scores, is_suspicious, model_details, rules_flags, filename="fraud_analysis"):
    try:
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fraud Detection Report - {filename}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .header {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }}
        .metric-value {{
            color: #667eea;
            font-size: 2rem;
            font-weight: 800;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: rgba(102, 126, 234, 0.05);
        }}
        .risk-high {{
            background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        .risk-medium {{
            background: linear-gradient(135deg, #f093fb 0%, #667eea 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        .risk-low {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 700;
        }}
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Fraud Detection Report</h1>
        <p>Intelligent Financial Fraud Detection System</p>
        <p>File: {filename}</p>
    </div>
    
    <div class="summary">
        <div class="metric-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Suspicious Cases</div>
            <div class="metric-value">{int(np.sum(is_suspicious)):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fraud Rate</div>
            <div class="metric-value">{(np.sum(is_suspicious) / len(df) * 100):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Fraud Score</div>
            <div class="metric-value">{np.mean(fraud_scores):.4f}</div>
        </div>
    </div>
    
    <h2>📊 Top Suspicious Transactions</h2>
    <table>
        <thead>
            <tr>
                <th>Transaction ID</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Fraud Score</th>
                <th>Risk Level</th>
            </tr>
        </thead>
        <tbody>
"""
        
        suspicious_indices = np.where(is_suspicious)[0]
        top_suspicious = sorted(suspicious_indices, key=lambda x: fraud_scores[x], reverse=True)[:20]
        
        for idx in top_suspicious:
            risk_score = fraud_scores[idx]
            if risk_score > 0.7:
                risk_class = "risk-high"
                risk_text = "High"
            elif risk_score > 0.4:
                risk_class = "risk-medium"
                risk_text = "Medium"
            else:
                risk_class = "risk-low"
                risk_text = "Low"
            
            html_template += f"""
            <tr>
                <td>{idx}</td>
                <td>{df.iloc[idx]['type'] if 'type' in df.columns else 'N/A'}</td>
                <td>{df.iloc[idx]['amount'] if 'amount' in df.columns else 0:,.2f}</td>
                <td>{risk_score:.4f}</td>
                <td><span class="{risk_class}">{risk_text}</span></td>
            </tr>
"""
        
        html_template += """
        </tbody>
    </table>
    
    <h2>🤖 Model Performance</h2>
"""
        
        if model_details:
            html_template += """
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Weight</th>
                <th>Anomalies Detected</th>
                <th>Avg Score</th>
            </tr>
        </thead>
        <tbody>
"""
            
            for model_name, details in model_details.items():
                html_template += f"""
            <tr>
                <td>{model_name.capitalize()}</td>
                <td>{details.get('weight', 0):.2f}</td>
                <td>{int(details.get('anomaly_count', 0))}</td>
                <td>{details.get('mean_score', 0):.4f}</td>
            </tr>
"""
            
            html_template += """
        </tbody>
    </table>
"""
        
        html_template += """
    <div class="footer">
        <p>Generated by FraudDetect 2.0 - Intelligent Financial Fraud Detection System</p>
        <p>© 2025 ClearFlow Security. All rights reserved.</p>
    </div>
</body>
</html>
"""
        
        return html_template
    except Exception as e:
        return f"<html><body><h1>Error generating report: {str(e)}</h1></body></html>"

def export_json_summary(df, fraud_scores, is_suspicious, model_details, rules_flags):
    try:
        summary = {
            "dataset_info": {
                "total_transactions": len(df),
                "suspicious_count": int(np.sum(is_suspicious)),
                "fraud_rate": float(np.sum(is_suspicious) / len(df)),
                "avg_fraud_score": float(np.mean(fraud_scores))
            },
            "model_performance": {},
            "top_suspicious_transactions": []
        }
        
        if model_details:
            summary["model_performance"] = {
                model_name: {
                    "weight": float(details.get('weight', 0)),
                    "anomalies_detected": int(details.get('anomaly_count', 0)),
                    "avg_score": float(details.get('mean_score', 0))
                }
                for model_name, details in model_details.items()
            }
        
        suspicious_indices = np.where(is_suspicious)[0]
        top_suspicious = sorted(suspicious_indices, key=lambda x: fraud_scores[x], reverse=True)[:10]
        
        for idx in top_suspicious:
            transaction_info = {
                "index": int(idx),
                "fraud_score": float(fraud_scores[idx]),
                "amount": float(df.iloc[idx]['amount']) if 'amount' in df.columns else 0
            }
            if 'type' in df.columns:
                transaction_info['type'] = df.iloc[idx]['type']
            summary["top_suspicious_transactions"].append(transaction_info)
        
        return json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to generate JSON summary: {str(e)}"}, indent=2)

def export_csv_results(df, fraud_scores, is_suspicious, explanations=None):
    try:
        result_df = df.copy()
        result_df['fraud_score'] = fraud_scores
        result_df['is_suspicious'] = is_suspicious
        
        if explanations:
            result_df['explanation'] = explanations
        else:
            result_df['explanation'] = "No explanation available"
        
        return result_df.to_csv(index=False)
    except Exception as e:
        return f"Error exporting CSV: {str(e)}"