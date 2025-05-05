# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import datetime

path = 'C:\\Users\\Mohammed Nawar\\Downloads\\Quantium ( Data Analysis )\\Quantium - Task 1\\'

# Load the transaction and customer data
transaction_data = pd.read_excel(path + 'QVI_transaction_data.xlsx')
customer_data = pd.read_csv(path + 'QVI_purchase_behaviour.csv')

# Exploratory Data Analysis (EDA)
#print(transaction_data.info())
#print(transaction_data.head())

#print(customer_data.info())
#print(customer_data.head())

# Convert the DATE column to datetime format by Microsoft Excel

# Ensure we are looking at chips
#print(transaction_data['PROD_NAME'].unique())


# Filter to remove non-chip products like salsa
transaction_data = transaction_data[~transaction_data['PROD_NAME'].str.contains('salsa', case=False)]

#print(transaction_data['PROD_NAME'])

# Summarize the data to check for nulls and possible outliers
print(transaction_data.describe())
print(transaction_data.isnull().sum())

# Remove Outliers values using certain threshold
print(transaction_data[transaction_data['PROD_QTY'] > 100])
transaction_data = transaction_data[transaction_data['PROD_QTY'] <= 100]

# Count transactions by date to identify missing dates
transaction_counts = transaction_data.groupby('DATE').size()
transaction_counts.plot(title="Transactions over time")
plt.xlabel("Date")
plt.ylabel("Transaction Count")
plt.show()
print(transaction_counts)

import pandas as pd
import matplotlib.pyplot as plt

dd = transaction_data.copy()

# تحويل عمود التاريخ إلى datetime
dd['DATE'] = pd.to_datetime(transaction_data['DATE'])

# تجميع عدد المعاملات أسبوعيًا
weekly_transactions = dd.groupby(pd.Grouper(key='DATE', freq='W'))['TOT_SALES'].count()

# رسم الخط الزمني
plt.figure(figsize=(12, 6))
plt.plot(weekly_transactions, marker='o', linestyle='-', label="Weekly Transactions", color='royalblue')

# إبراز الأسبوع الذي يسبق الكريسماس
highlight_date = weekly_transactions.index[weekly_transactions.index.month == 12][-2]  # الأسبوع قبل الكريسماس
highlight_value = weekly_transactions.loc[highlight_date]

plt.scatter(highlight_date, highlight_value, color='black', zorder=3)  
plt.text(highlight_date, highlight_value - 200, "Notable increase before Christmas", fontsize=10, ha='center')

# تنسيق الرسم البياني
plt.title("Snack Food - Chips: Weekly Transactions Over Time", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number of Transactions", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# Feature Engineering: Create PACK_SIZE and BRAND columns
transaction_data['PACK_SIZE'] = transaction_data['PROD_NAME'].str.extract(r'(\d+)').astype(int)
transaction_data['BRAND'] = transaction_data['PROD_NAME'].str.split().str[0]


#### Clean brand names
transaction_data['BRAND'] = transaction_data['BRAND'].replace({
    'RRD': 'RRD',
    'Red': 'RRD',
    'Snbts': 'SUNBITES', 
    'Infzns': 'INFUZIONS',
    
    'WW': 'WOOLWORTHS', 
    'SMITH': 'SMITHS',
    'NCC': 'NATURAL', 
    'DORITO': 'DORITOS',
    'GRAIN': 'GRNWVES'
})


# Examine the cleaned data
print(transaction_data['PACK_SIZE'].value_counts())
print(transaction_data['BRAND'].value_counts())

# Number of Transactions by Pack Size
plt.figure(figsize=(10, 6))
sns.countplot(x='PACK_SIZE', data=transaction_data, palette='viridis', order=transaction_data['PACK_SIZE'].value_counts().index)
plt.title('Number of Transactions by Pack Size', fontsize=14)
plt.xlabel('Pack Size (g)', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



#  customer data

# 1. Summarize key columns
print("Summary of Customer Data:")
print(customer_data.describe())

# 2. Plot distribution of LIFESTAGE
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_data, x='LIFESTAGE', order=customer_data['LIFESTAGE'].value_counts().index, palette='viridis')
plt.title('Distribution of Lifestage', fontsize=16)
plt.xlabel('Lifestage', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()

# 3. Plot distribution of PREMIUM_CUSTOMER
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_data, x='PREMIUM_CUSTOMER', order=customer_data['PREMIUM_CUSTOMER'].value_counts().index, palette='muted')
plt.title('Distribution of Premium Customer', fontsize=16)
plt.xlabel('Premium Customer Segment', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()

###Average number of units per customer by LIFESTAGE and PREMIUM_CUSTOMER
# 4. Combine 2 column and compare
customer_lifestage_premium = customer_data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).size().unstack()
customer_lifestage_premium.plot(kind='bar', stacked=False, figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Lifestage vs Premium Customer Distribution', fontsize=16)
plt.xlabel('Lifestage', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Premium Customer Segment')
plt.tight_layout()
plt.show()


# Merge the transaction and customer datasets
data = pd.merge(transaction_data, customer_data, how='left', on='LYLTY_CARD_NBR')

# Check for missing customer details
print(data.isnull().sum())           # No missing values



                      # ( Data Analysis ) #

# Total sales by LIFESTAGE and PREMIUM_CUSTOMER
sales_by_segment = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum()
sales_by_segment.unstack().plot(kind='bar', stacked=False, figsize=(10, 6))
plt.title("Total Sales by Customer Segment")
plt.xlabel("Lifestage")
plt.ylabel("Total Sales")
plt.legend(title="Premium Customer")
plt.show()


# Total Number of Customer by LIFESTAGE and PREMIUM_CUSTOMER
sales_by_segment = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].nunique()
sales_by_segment.unstack().plot(kind='bar', stacked=False, figsize=(10, 6))
plt.title("Number of Customers by LIFESTAGE and PREMIUM_CUSTOMER")
plt.xlabel("Lifestage")
plt.ylabel("Number of Customer")
plt.legend(title="Premium Customer")
plt.show()

# Average price per unit by LIFESTAGE and PREMIUM_CUSTOMER
data['PRICE_PER_UNIT'] = data['TOT_SALES'] / data['PROD_QTY']
avg_price_by_segment = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PRICE_PER_UNIT'].mean()
avg_price_by_segment.unstack().plot(kind='bar', figsize=(10, 6))
plt.title("Average Price Per Unit by Customer Segment")
plt.xlabel("Lifestage")
plt.ylabel("Average Unit Price Per Transection")
plt.legend(title="Premium Customer")
plt.show()


print('###########################################################')





####### ########### #########
# Statistical Test: Compare Mainstream VS Premium and Budget  -> ( midage and young singles/couples )

mainstream_midage_young = data[
    (data['LIFESTAGE'].isin(['MIDAGE SINGLES/COUPLES', 'YOUNG SINGLES/COUPLES'])) &
    (data['PREMIUM_CUSTOMER'] == 'Mainstream')]['PRICE_PER_UNIT']

budget_midage_young = data[
    (data['LIFESTAGE'].isin(['MIDAGE SINGLES/COUPLES', 'YOUNG SINGLES/COUPLES'])) &
    (data['PREMIUM_CUSTOMER'] == 'Budget')]['PRICE_PER_UNIT']

premium_midage_young = data[
    (data['LIFESTAGE'].isin(['MIDAGE SINGLES/COUPLES', 'YOUNG SINGLES/COUPLES'])) &
    (data['PREMIUM_CUSTOMER'] == 'Premium')]['PRICE_PER_UNIT']

# t-test -> mainstream , budget
t_stat1, p_value1 = ttest_ind(mainstream_midage_young, budget_midage_young, equal_var=False)
# t-test -> mainstream , premium
t_stat2, p_value2 = ttest_ind(mainstream_midage_young, premium_midage_young, equal_var=False)

print(f"T-test between Mainstream and Budget: p-value = {p_value1}")
print(f"T-test between Mainstream and Premium: p-value = {p_value2}")

if p_value1 < 0.05: print("The unit price for mainstream midage and young singles/couples is significantly higher than that of budget.")
else: print("The unit price for mainstream midage and young singles/couples is NOT significantly higher than that of budget.")

if p_value2 < 0.05: print("The unit price for mainstream midage and young singles/couples is significantly higher than that of premium.")
else: print("The unit price for mainstream midage and young singles/couples is NOT significantly higher than that of premium.")


# Deep Dive: Preferred brands by target segment
mainstream_young = data[(data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & 
                        (data['PREMIUM_CUSTOMER'] == 'Mainstream')]
brand_preferences = mainstream_young['BRAND'].value_counts()
brand_preferences.plot(kind='bar', figsize=(10, 6))
plt.title("Preferred Brands by Mainstream Young Singles/Couples")
plt.xlabel("Brand")
plt.ylabel("Frequency")
plt.show()


###################

from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Filter data for "Mainstream - Young Singles/Couples"
mainstream_young = data[(data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') &
                        (data['PREMIUM_CUSTOMER'] == 'Mainstream')]

# Step 2: Prepare data in transactional format
# Create a basket with each transaction as a row and brands as columns
basket = mainstream_young.groupby(['LYLTY_CARD_NBR', 'BRAND'])['TOT_SALES'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert to binary format (1 if purchased, 0 otherwise)

# Step 3: Apply the Apriori Algorithm
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Calculate num_itemsets
num_itemsets = frequent_itemsets['support'].count()

# Step 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=num_itemsets)


# Step 5: Filter rules for insights
# Sort by lift for strong associations
rules = rules.sort_values(by='lift', ascending=False)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Optional: Visualize top associations
top_rules = rules.head(10)
top_rules['rule'] = top_rules['antecedents'].astype(str) + " -> " + top_rules['consequents'].astype(str)

plt.figure(figsize=(10, 6))
plt.barh(top_rules['rule'], top_rules['lift'], color='skyblue')
plt.xlabel('Lift')
plt.title('Top 10 Association Rules by Lift')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Filter data for the target segment
mainstream_young = data[
    (data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & 
    (data['PREMIUM_CUSTOMER'] == 'Mainstream')
]

# Filter data for the rest of the population
rest_of_population = data[
    ~((data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & 
      (data['PREMIUM_CUSTOMER'] == 'Mainstream'))
]

# Plot pack size distribution for the target segment
plt.figure(figsize=(12, 6))
sns.kdeplot(mainstream_young['PACK_SIZE'], label='Mainstream - Young Singles/Couples', shade=True, color='blue')
sns.kdeplot(rest_of_population['PACK_SIZE'], label='Rest of the Population', shade=True, color='orange')
plt.title("Pack Size Preference: Mainstream Young Singles/Couples vs Rest of the Population", fontsize=16)
plt.xlabel("Pack Size (g)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate summary statistics for pack size
mainstream_pack_size_mean = mainstream_young['PACK_SIZE'].mean()
rest_pack_size_mean = rest_of_population['PACK_SIZE'].mean()

print(f"Average pack size for Mainstream - Young Singles/Couples: {mainstream_pack_size_mean:.2f}g")
print(f"Average pack size for the Rest of the Population: {rest_pack_size_mean:.2f}g")



#transaction_data.to_csv(path+'1 dada transaction_data.csv', index=False)
#data.to_csv(path+'merge_data.csv', index=False)

