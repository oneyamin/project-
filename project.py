import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

def clean_price(price_str):
    """Convert price string to numeric value"""
    if pd.isna(price_str):
        return np.nan
    # Remove $ and commas, then convert to float
    cleaned = re.sub(r'[$,]', '', str(price_str))
    try:
        return float(cleaned)
    except:
        return np.nan

def clean_mileage(mileage_str):
    """Convert mileage string to numeric value"""
    if pd.isna(mileage_str):
        return np.nan
    # Extract numeric part from mileage string
    cleaned = re.sub(r'[,\s]', '', str(mileage_str))
    numbers = re.findall(r'\d+', cleaned)
    if numbers:
        return float(numbers[0])
    return np.nan

def calculate_depreciation_rate(price, age):
    """Calculate annual depreciation rate"""
    if age == 0:
        return 0
    # Assuming original MSRP is roughly 1.5x current price for newer cars
    # This is a simplified estimation
    estimated_original = price * (1 + 0.15 * age)
    depreciation_rate = ((estimated_original - price) / estimated_original) / age * 100
    return max(0, min(50, depreciation_rate))  # Cap between 0-50% per year

def main():
    print("Car Price Depreciation Analysis")
    print("=" * 40)
    
    # Load the dataset
    try:
        df = pd.read_csv('attached_assets/used_cars_1753641342195.csv')
        print(f"Dataset loaded successfully: {len(df)} cars")
    except FileNotFoundError:
        print("Error: Could not find the car dataset file")
        return
    
    # Clean the data
    df['price_numeric'] = df['price'].apply(clean_price)
    df['mileage_numeric'] = df['milage'].apply(clean_mileage)  # Note: 'milage' in original CSV
    
    # Calculate car age (current year - model year)
    current_year = datetime.now().year
    df['age'] = current_year - df['model_year']
    
    # Remove invalid data
    df_clean = df.dropna(subset=['price_numeric', 'model_year', 'age'])
    df_clean = df_clean[df_clean['age'] >= 0]  # Remove future cars
    df_clean = df_clean[df_clean['price_numeric'] > 1000]  # Remove unrealistic prices
    
    print(f"Clean dataset: {len(df_clean)} cars")
    
    # Calculate depreciation metrics
    df_clean['depreciation_rate'] = df_clean.apply(
        lambda row: calculate_depreciation_rate(row['price_numeric'], row['age']), 
        axis=1
    )
    
    # Analysis 1: Average depreciation by age
    print("\n1. Average Car Price by Age:")
    print("-" * 30)
    age_analysis = df_clean.groupby('age').agg({
        'price_numeric': ['mean', 'median', 'count']
    }).round(2)
    age_analysis.columns = ['Mean_Price', 'Median_Price', 'Count']
    print(age_analysis.head(10))
    
    # Analysis 2: Brand depreciation comparison
    print("\n2. Average Price by Brand (Top 10 brands):")
    print("-" * 40)
    brand_analysis = df_clean.groupby('brand').agg({
        'price_numeric': ['mean', 'count'],
        'age': 'mean'
    }).round(2)
    brand_analysis.columns = ['Avg_Price', 'Count', 'Avg_Age']
    brand_analysis = brand_analysis[brand_analysis['Count'] >= 5]  # At least 5 cars
    brand_analysis_sorted = brand_analysis.sort_values('Avg_Price', ascending=False)
    print(brand_analysis_sorted.head(10))
    
    # Analysis 3: Mileage impact on price
    print("\n3. Price vs Mileage Analysis:")
    print("-" * 30)
    df_with_mileage = df_clean.dropna(subset=['mileage_numeric'])
    correlation = df_with_mileage['price_numeric'].corr(df_with_mileage['mileage_numeric'])
    print(f"Price-Mileage Correlation: {correlation:.3f}")
    
    # Mileage bins analysis
    df_with_mileage['mileage_bin'] = pd.cut(
        df_with_mileage['mileage_numeric'], 
        bins=[0, 30000, 60000, 100000, 200000, float('inf')],
        labels=['0-30k', '30-60k', '60-100k', '100-200k', '200k+']
    )
    mileage_analysis = df_with_mileage.groupby('mileage_bin')['price_numeric'].agg(['mean', 'count']).round(2)
    print("\nAverage Price by Mileage Range:")
    print(mileage_analysis)
    
    # Analysis 4: Luxury vs Regular brands
    print("\n4. Luxury vs Regular Brand Analysis:")
    print("-" * 35)
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Ferrari', 
                     'Lamborghini', 'Bentley', 'Rolls-Royce', 'Aston', 'Maserati']
    
    df_clean['is_luxury'] = df_clean['brand'].isin(luxury_brands)
    luxury_analysis = df_clean.groupby('is_luxury').agg({
        'price_numeric': ['mean', 'median'],
        'age': 'mean'
    }).round(2)
    luxury_analysis.columns = ['Mean_Price', 'Median_Price', 'Avg_Age']
    luxury_analysis.index = ['Regular', 'Luxury']
    print(luxury_analysis)
    
    # Analysis 5: Depreciation by fuel type
    print("\n5. Price Analysis by Fuel Type:")
    print("-" * 30)
    fuel_analysis = df_clean.groupby('fuel_type').agg({
        'price_numeric': ['mean', 'count'],
        'age': 'mean'
    }).round(2)
    fuel_analysis.columns = ['Avg_Price', 'Count', 'Avg_Age']
    fuel_analysis = fuel_analysis[fuel_analysis['Count'] >= 10]  # At least 10 cars
    fuel_analysis_sorted = fuel_analysis.sort_values('Avg_Price', ascending=False)
    print(fuel_analysis_sorted)
    
    # Key insights
    print("\n" + "="*50)
    print("KEY DEPRECIATION INSIGHTS:")
    print("="*50)
    
    # Calculate average depreciation per year
    yearly_depreciation = df_clean[df_clean['age'] <= 10].groupby('age')['price_numeric'].mean()
    if len(yearly_depreciation) > 1:
        first_year_price = yearly_depreciation.iloc[0] if 0 in yearly_depreciation.index else yearly_depreciation.iloc[0]
        fifth_year_price = yearly_depreciation.iloc[min(5, len(yearly_depreciation)-1)]
        depreciation_5_years = ((first_year_price - fifth_year_price) / first_year_price) * 100
        print(f"• Average 5-year depreciation: {depreciation_5_years:.1f}%")
    
    # Most valuable brands
    top_brands = brand_analysis_sorted.head(3)
    print(f"• Highest value brands: {', '.join(top_brands.index[:3])}")
    
    # Mileage impact
    low_mileage = df_with_mileage[df_with_mileage['mileage_numeric'] < 30000]['price_numeric'].mean()
    high_mileage = df_with_mileage[df_with_mileage['mileage_numeric'] > 100000]['price_numeric'].mean()
    mileage_impact = ((low_mileage - high_mileage) / low_mileage) * 100
    print(f"• High mileage impact: {mileage_impact:.1f}% price reduction")
    
    # Luxury premium
    if len(luxury_analysis) == 2:
        luxury_premium = ((luxury_analysis.loc['Luxury', 'Mean_Price'] - 
                          luxury_analysis.loc['Regular', 'Mean_Price']) / 
                         luxury_analysis.loc['Regular', 'Mean_Price']) * 100
        print(f"• Luxury brand premium: {luxury_premium:.1f}% higher prices")
    
    # Create visualizations
    create_visualizations(df_clean, age_analysis, brand_analysis_sorted, luxury_analysis)
    
    print("\nAnalysis complete! This data shows how various factors affect car depreciation.")

def create_visualizations(df_clean, age_analysis, brand_analysis, luxury_analysis):
    """Create comprehensive visualizations for the depreciation analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Depreciation by Age
    plt.subplot(3, 3, 1)
    age_data = age_analysis.reset_index()
    plt.plot(age_data['age'], age_data['Mean_Price'], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.title('Average Car Price by Age', fontsize=14, fontweight='bold')
    plt.xlabel('Car Age (Years)')
    plt.ylabel('Average Price ($)')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Top Brands by Price
    plt.subplot(3, 3, 2)
    top_brands = brand_analysis.head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_brands)))
    bars = plt.bar(range(len(top_brands)), top_brands['Avg_Price'], color=colors)
    plt.title('Top 10 Brands by Average Price', fontsize=14, fontweight='bold')
    plt.xlabel('Brand')
    plt.ylabel('Average Price ($)')
    plt.xticks(range(len(top_brands)), top_brands.index, rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 3. Price Distribution
    plt.subplot(3, 3, 3)
    plt.hist(df_clean['price_numeric'], bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
    plt.title('Price Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 4. Mileage vs Price Scatter
    plt.subplot(3, 3, 4)
    df_sample = df_clean.dropna(subset=['mileage_numeric']).sample(min(1000, len(df_clean)))
    plt.scatter(df_sample['mileage_numeric'], df_sample['price_numeric'], 
                alpha=0.6, color='#F18F01', s=20)
    plt.title('Price vs Mileage', fontsize=14, fontweight='bold')
    plt.xlabel('Mileage')
    plt.ylabel('Price ($)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # 5. Luxury vs Regular Brands
    plt.subplot(3, 3, 5)
    categories = ['Regular', 'Luxury']
    prices = [luxury_analysis.loc['Regular', 'Mean_Price'], 
              luxury_analysis.loc['Luxury', 'Mean_Price']]
    colors = ['#C73E1D', '#3A86FF']
    bars = plt.bar(categories, prices, color=colors, alpha=0.8)
    plt.title('Luxury vs Regular Brand Pricing', fontsize=14, fontweight='bold')
    plt.ylabel('Average Price ($)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    for bar, price in zip(bars, prices):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'${price:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Age Distribution
    plt.subplot(3, 3, 6)
    plt.hist(df_clean['age'], bins=30, alpha=0.7, color='#8338EC', edgecolor='black')
    plt.title('Age Distribution of Cars', fontsize=14, fontweight='bold')
    plt.xlabel('Age (Years)')
    plt.ylabel('Number of Cars')
    
    # 7. Fuel Type Analysis
    plt.subplot(3, 3, 7)
    fuel_counts = df_clean['fuel_type'].value_counts().head(6)
    plt.pie(fuel_counts.values, labels=fuel_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=plt.cm.Set3.colors)
    plt.title('Distribution by Fuel Type', fontsize=14, fontweight='bold')
    
    # 8. Depreciation Rate by Brand Category
    plt.subplot(3, 3, 8)
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche']
    luxury_data = df_clean[df_clean['brand'].isin(luxury_brands)]
    regular_data = df_clean[~df_clean['brand'].isin(luxury_brands)]
    
    if len(luxury_data) > 0 and len(regular_data) > 0:
        luxury_age_price = luxury_data.groupby('age')['price_numeric'].mean()
        regular_age_price = regular_data.groupby('age')['price_numeric'].mean()
        
        ages = sorted(set(luxury_age_price.index) & set(regular_age_price.index))
        if ages:
            plt.plot(ages, [luxury_age_price[age] for age in ages], 
                    marker='o', label='Luxury Brands', linewidth=2)
            plt.plot(ages, [regular_age_price[age] for age in ages], 
                    marker='s', label='Regular Brands', linewidth=2)
            plt.title('Depreciation: Luxury vs Regular', fontsize=14, fontweight='bold')
            plt.xlabel('Age (Years)')
            plt.ylabel('Average Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 9. Price vs Age Scatter with Trend
    plt.subplot(3, 3, 9)
    sample_data = df_clean.sample(min(1500, len(df_clean)))
    plt.scatter(sample_data['age'], sample_data['price_numeric'], 
                alpha=0.5, color='#FB8500', s=15)
    
    # Add trend line
    z = np.polyfit(sample_data['age'], sample_data['price_numeric'], 1)
    p = np.poly1d(z)
    plt.plot(sample_data['age'].sort_values(), p(sample_data['age'].sort_values()), 
             "r--", alpha=0.8, linewidth=2)
    
    plt.title('Price vs Age with Trend Line', fontsize=14, fontweight='bold')
    plt.xlabel('Age (Years)')
    plt.ylabel('Price ($)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout(pad=3.0)
    plt.savefig('car_depreciation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis summary
    print("\n" + "="*60)
    print("DETAILED MARKET INSIGHTS:")
    print("="*60)
    
    # Best value retention brands
    recent_cars = df_clean[df_clean['age'] <= 5]
    if len(recent_cars) > 0:
        brand_retention = recent_cars.groupby('brand').agg({
            'price_numeric': 'mean',
            'age': 'mean'
        }).round(2)
        brand_retention = brand_retention[brand_retention.index.map(
            lambda x: len(recent_cars[recent_cars['brand'] == x]) >= 10
        )]
        if len(brand_retention) > 0:
            top_retention = brand_retention.sort_values('price_numeric', ascending=False).head(5)
            print("\n• Best Value Retention (Recent Models):")
            for brand, data in top_retention.iterrows():
                print(f"  {brand}: ${data['price_numeric']:,.0f} avg (age {data['age']:.1f})")
    
    # Depreciation rate analysis
    old_cars = df_clean[df_clean['age'] >= 10]
    new_cars = df_clean[df_clean['age'] <= 3]
    
    if len(old_cars) > 0 and len(new_cars) > 0:
        old_avg = old_cars['price_numeric'].mean()
        new_avg = new_cars['price_numeric'].mean()
        total_depreciation = ((new_avg - old_avg) / new_avg) * 100
        print(f"\n• Total Market Depreciation (10+ years): {total_depreciation:.1f}%")
    
    # Sweet spot analysis
    sweet_spot = df_clean[(df_clean['age'] >= 3) & (df_clean['age'] <= 7)]
    if len(sweet_spot) > 0:
        sweet_price = sweet_spot['price_numeric'].mean()
        sweet_age = sweet_spot['age'].mean()
        print(f"• Sweet Spot for Buyers: {sweet_age:.1f} years old, ${sweet_price:,.0f} average")
    
    print(f"\n• Sample Size: {len(df_clean):,} vehicles analyzed")
    print(f"• Price Range: ${df_clean['price_numeric'].min():,.0f} - ${df_clean['price_numeric'].max():,.0f}")
    print(f"• Age Range: {df_clean['age'].min()} - {df_clean['age'].max()} years")

if _name_ == "_main_":
    main()