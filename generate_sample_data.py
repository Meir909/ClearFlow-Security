import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_data(num_transactions=10000, fraud_ratio=0.05):
    """
    Generate synthetic financial transaction data for testing
    
    Parameters:
    num_transactions: Number of transactions to generate
    fraud_ratio: Proportion of fraudulent transactions
    
    Returns:
    df: DataFrame with synthetic transaction data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate customer IDs
    num_customers = num_transactions // 10
    customer_ids = [f"C{random.randint(100000, 999999)}" for _ in range(num_customers)]
    
    # Generate destination IDs
    num_destinations = num_transactions // 5
    destination_ids = [f"D{random.randint(100000, 999999)}" for _ in range(num_destinations)]
    
    # Transaction types
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
    
    # Generate transactions
    transactions = []
    
    for i in range(num_transactions):
        # Basic transaction info
        step = random.randint(1, 720)  # Up to 30 days (assuming 1 step = 1 hour)
        trans_type = random.choices(transaction_types, weights=[0.3, 0.4, 0.1, 0.1, 0.1])[0]
        
        # Customer info
        orig_customer = random.choice(customer_ids)
        dest_customer = random.choice(destination_ids)
        
        # Balance info (simplified)
        orig_balance = max(0, np.random.lognormal(8, 1.5))  # Log-normal distribution for balances
        dest_balance = max(0, np.random.lognormal(7, 1.2))
        
        # Amount (dependent on balances for realistic transactions)
        if trans_type in ['CASH_OUT', 'TRANSFER']:
            max_amount = min(orig_balance * 0.8, 100000)  # Cap at 80% of balance or 100k
            amount = random.uniform(1, max(max_amount, 1))
        elif trans_type == 'CASH_IN':
            amount = random.uniform(1, min(dest_balance * 0.5, 50000))
        else:  # PAYMENT, DEBIT
            amount = random.uniform(1, min(orig_balance * 0.3, 20000))
        
        # Update balances
        if trans_type in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
            new_orig_balance = max(0, orig_balance - amount)
            new_dest_balance = dest_balance + amount
        else:  # CASH_IN
            new_orig_balance = orig_balance + amount
            new_dest_balance = max(0, dest_balance - amount)
        
        # Determine if fraudulent (with some logic)
        is_fraud = 0
        is_flagged_fraud = 0
        
        # Increase probability of fraud based on certain conditions
        fraud_probability = 0.01  # Base probability
        
        # High amount transactions more likely to be fraudulent
        if amount > 50000:
            fraud_probability += 0.1
        elif amount > 10000:
            fraud_probability += 0.05
            
        # New destination accounts more likely to be fraudulent
        if random.random() < 0.1:  # 10% chance of being a new destination
            fraud_probability += 0.05
            
        # Very low balance after transaction
        if new_orig_balance < 10:
            fraud_probability += 0.03
            
        # Unusual hours (1-5 AM)
        hour = step % 24
        if 1 <= hour <= 5:
            fraud_probability += 0.02
            
        # Round amounts
        if amount in [1000, 5000, 10000, 20000, 50000]:
            fraud_probability += 0.03
        
        # Randomly assign fraud based on probability
        if random.random() < fraud_probability:
            is_fraud = 1
            # Flagged fraud is a subset of actual fraud
            if random.random() < 0.3:  # 30% of fraud gets flagged
                is_flagged_fraud = 1
        
        # Force some fraud to meet the target ratio
        if i < num_transactions * fraud_ratio * 0.5:  # Ensure minimum fraud
            is_fraud = 1
            if random.random() < 0.3:
                is_flagged_fraud = 1
        
        transactions.append({
            'step': step,
            'type': trans_type,
            'amount': round(amount, 2),
            'nameOrig': orig_customer,
            'oldbalanceOrg': round(orig_balance, 2),
            'newbalanceOrig': round(new_orig_balance, 2),
            'nameDest': dest_customer,
            'oldbalanceDest': round(dest_balance, 2),
            'newbalanceDest': round(new_dest_balance, 2),
            'isFraud': is_fraud,
            'isFlaggedFraud': is_flagged_fraud
        })
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample financial transaction data...")
    df = generate_sample_data(num_transactions=10000, fraud_ratio=0.05)
    
    # Save to CSV
    df.to_csv("sample_financial_data.csv", index=False)
    
    # Print summary
    print(f"\nGenerated {len(df)} transactions")
    print(f"Fraudulent transactions: {df['isFraud'].sum()} ({df['isFraud'].mean():.2%})")
    print(f"Flagged fraudulent transactions: {df['isFlaggedFraud'].sum()} ({df['isFlaggedFraud'].mean():.2%})")
    
    # Print first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData saved to 'sample_financial_data.csv'")