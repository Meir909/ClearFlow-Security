import pandas as pd
import numpy as np

# Create a test dataset with some problematic data to test our fixes
data = {
    'step': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'CASH_IN', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT'],
    'amount': [100.50, 2500.00, 75.25, 89.99, 5000.00, 1000.00, 150.00, 2000.00, 50.00, 125.75],
    'nameOrig': ['C123456789', 'C987654321', 'C123456789', 'C987654321', 'C111111111', 'C987654321', 'C123456789', 'C987654321', 'C123456789', 'C987654321'],
    'oldbalanceOrg': [1500.00, 10000.00, 1400.00, 2500.00, 30000.00, 5000.00, 1250.00, 8000.00, 500.00, 2000.00],
    'newbalanceOrig': [1399.50, 7500.00, 1324.75, 2410.01, 25000.00, 6000.00, 1100.00, 6000.00, 450.00, 1874.25],
    'nameDest': ['MERCHANT1', 'C123456789', 'ATM001', 'SHOP001', 'C222222222', 'BANK001', 'CAFE001', 'C333333333', 'GAS_STATION', 'STORE001'],
    'oldbalanceDest': [0.00, 1000.00, 500.00, 100.00, 1000.00, 0.00, 200.00, 100.00, 300.00, 50.00],
    'newbalanceDest': [100.50, 6000.00, 575.25, 189.99, 26000.00, 1000.00, 350.00, 300.00, 350.00, 175.75]
}

# Intentionally add some problematic data
data['amount'][2] = 'N/A'  # Text in numeric field
data['oldbalanceOrg'][5] = ''  # Empty value
data['newbalanceOrig'][7] = 'invalid'  # Invalid text

df = pd.DataFrame(data)
df.to_csv('test_problematic_data.csv', index=False)
print("Test file with problematic data 'test_problematic_data.csv' created successfully!")
print("\nData preview:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)