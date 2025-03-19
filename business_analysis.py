import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('ggplot')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

import os
if not os.path.exists('results'):
    os.makedirs('results')

def import_datasets():
    """Imports the required datasets for analysis"""
    try:
        purchase_items = pd.read_csv('data/Order_Details_19795F61CF.csv')
        revenue_goals = pd.read_csv('data/Sales_target_DD2E9B96A0.csv')
        purchase_records = pd.read_csv('data/List_of_Orders_55FFC79CF8.csv')
        
        print("All datasets successfully imported!")
        
        print("\n--- Purchase Items Dataset ---")
        print(f"Dimensions: {purchase_items.shape}")
        print(purchase_items.head(3))
        print(purchase_items.dtypes)
        
        print("\n--- Revenue Goals Dataset ---")
        print(f"Dimensions: {revenue_goals.shape}")
        print(revenue_goals.head(3))
        print(revenue_goals.dtypes)
        
        print("\n--- Purchase Records Dataset ---")
        print(f"Dimensions: {purchase_records.shape}")
        print(purchase_records.head(3))
        print(purchase_records.dtypes)
        
        return purchase_items, revenue_goals, purchase_records
        
    except Exception as e:
        print(f"Error encountered while importing datasets: {e}")
        return None, None, None

# Question 1: Sales Analysis
# Part 1) Sales and Profitability Analysis
def examine_sales_profit(purchase_items, purchase_records):
    print("\n========== PART 1: SALES AND PROFITABILITY ANALYSIS ==========\n")
    
    combined_dataset = pd.merge(purchase_records, purchase_items, on='Order ID', how='inner')
    print(f"Combined dataset dimensions: {combined_dataset.shape}")
    print(combined_dataset.head(3))
    
    product_group_revenue = combined_dataset.groupby('Category')['Amount'].sum().reset_index()
    product_group_revenue = product_group_revenue.sort_values('Amount', ascending=False)
    
    product_group_avg_earning = combined_dataset.groupby(['Category', 'Order ID'])['Profit'].mean().reset_index()
    product_group_avg_earning = product_group_avg_earning.groupby('Category')['Profit'].mean().reset_index()
    product_group_avg_earning.columns = ['Category', 'Avg Profit per Order']
    
    product_group_metrics = combined_dataset.groupby('Category').agg({
        'Profit': 'sum',
        'Amount': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    product_group_metrics['Profit Margin (%)'] = (product_group_metrics['Profit'] / product_group_metrics['Amount'] * 100)
    product_group_metrics.rename(columns={'Order ID': 'Order Count'}, inplace=True)
    
    product_group_summary = pd.merge(product_group_revenue, product_group_avg_earning, on='Category')
    product_group_summary = pd.merge(
        product_group_summary, 
        product_group_metrics[['Category', 'Profit', 'Profit Margin (%)', 'Order Count']], 
        on='Category'
    )
    product_group_summary = product_group_summary.rename(columns={
        'Amount': 'Total Sales',
        'Profit': 'Total Profit'
    })
    
    product_group_summary = product_group_summary.sort_values('Total Sales', ascending=False)
    
    print("\nProduct Category Performance Metrics:")
    print(product_group_summary)
    
    leading_category = product_group_summary.iloc[0]['Category']
    lagging_category = product_group_summary.iloc[-1]['Category']
    
    most_profitable_category = product_group_summary.loc[product_group_summary['Profit Margin (%)'].idxmax()]['Category']
    least_profitable_category = product_group_summary.loc[product_group_summary['Profit Margin (%)'].idxmin()]['Category']
    
    print(f"\nLeading category by sales: {leading_category}")
    print(f"Lagging category by sales: {lagging_category}")
    print(f"Most profitable category: {most_profitable_category}")
    print(f"Least profitable category: {least_profitable_category}")
    
    # Visualizations
    # 1. Total Sales by Category
    plt.figure(figsize=(10, 6))
    axis = sns.barplot(x='Category', y='Total Sales', hue='Category', data=product_group_summary, palette='viridis', legend=False)
    plt.title('Total Sales by Product Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=14)
    plt.ylabel('Total Sales (Amount)', fontsize=14)
    plt.xticks(rotation=0)
    
    for i, p in enumerate(axis.patches):
        axis.annotate(f'${p.get_height():,.0f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/total_sales_by_category.png', dpi=300)
    
    # 2. Profit Margin by Category
    plt.figure(figsize=(10, 6))
    axis = sns.barplot(x='Category', y='Profit Margin (%)', data=product_group_summary, palette='coolwarm')
    
    for i, p in enumerate(axis.patches):
        axis.annotate(f"{p.get_height():.1f}%",
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.title('Profit Margin by Product Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=14)
    plt.ylabel('Profit Margin (%)', fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/profit_margin_by_category.png', dpi=300)
    
    # 3. Average Profit per Order by Category
    plt.figure(figsize=(10, 6))
    axis = sns.barplot(x='Category', y='Avg Profit per Order', data=product_group_summary, palette='magma')
    
    for i, p in enumerate(axis.patches):
        axis.annotate(f'${p.get_height():,.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.title('Average Profit per Order by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Product Category', fontsize=14)
    plt.ylabel('Average Profit per Order', fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/avg_profit_by_category.png', dpi=300)
    
    # 4. Combined visualization: Total Sales, Profit, and Profit Margin
    figure, axis1 = plt.subplots(figsize=(14, 8))
    
    # Bar chart
    x_positions = np.arange(len(product_group_summary))
    bar_width = 0.35
    
    axis1.bar(x_positions - bar_width/2, product_group_summary['Total Sales'], bar_width, label='Total Sales', color='skyblue')
    axis1.bar(x_positions + bar_width/2, product_group_summary['Total Profit'], bar_width, label='Total Profit', color='lightgreen')
    
    axis1.set_xlabel('Product Category', fontsize=14)
    axis1.set_ylabel('Amount', fontsize=14)
    axis1.set_title('Sales, Profit, and Margin by Category', fontsize=16, fontweight='bold')
    axis1.set_xticks(x_positions)
    axis1.set_xticklabels(product_group_summary['Category'])
    axis1.legend(loc='upper left')
    
    axis2 = axis1.twinx()
    axis2.plot(x_positions, product_group_summary['Profit Margin (%)'], 'ro-', linewidth=2, label='Profit Margin (%)')
    axis2.set_ylabel('Profit Margin (%)', fontsize=14)
    axis2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/category_performance_combined.png', dpi=300)
    
    # 5. Sub-category analysis
    subcategory_metrics = combined_dataset.groupby(['Category', 'Sub-Category']).agg({
        'Amount': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    subcategory_metrics['Profit Margin (%)'] = (
        subcategory_metrics['Profit'] / subcategory_metrics['Amount'] * 100
    )
    
    subcategory_metrics = subcategory_metrics.rename(columns={
        'Amount': 'Total Sales',
        'Profit': 'Total Profit',
        'Order ID': 'Order Count'
    })
    
    subcategory_metrics = subcategory_metrics.sort_values(
        ['Category', 'Total Sales'], ascending=[True, False]
    )
    
    print("\nSub-Category Performance Metrics:")
    print(subcategory_metrics)
    
    # Visualize top 10 sub-categories
    top_subcategories = subcategory_metrics.sort_values('Total Sales', ascending=False).head(10)
    
    plt.figure(figsize=(14, 10))
    axis = sns.barplot(x='Total Sales', y='Sub-Category', hue='Category', 
                    data=top_subcategories, dodge=False, palette='Set2')
    
    for i, p in enumerate(axis.patches):
        axis.annotate(f'${p.get_width():,.0f}', 
                   (p.get_width(), p.get_y() + p.get_height() / 2),
                   ha = 'left', va = 'center', fontsize=10)
    
    plt.title('Top 10 Sub-Categories by Sales', fontsize=16, fontweight='bold')
    plt.xlabel('Total Sales', fontsize=14)
    plt.ylabel('Sub-Category', fontsize=14)
    plt.legend(title='Category')
    plt.tight_layout()
    plt.savefig('results/top_subcategories_by_sales.png', dpi=300)
    
    return combined_dataset, product_group_summary, subcategory_metrics

# Part 2) Target Achievement Analysis
def examine_target_achievement(revenue_goals):
    print("\n========== PART 2: TARGET ACHIEVEMENT ANALYSIS ==========\n")
    
    furniture_goals = revenue_goals[revenue_goals['Category'] == 'Furniture'].copy()

    furniture_goals['Month'] = pd.to_datetime(furniture_goals['Month of Order Date'], format='%b-%y')
    furniture_goals = furniture_goals.sort_values('Month')

    furniture_goals['MoM Change (%)'] = furniture_goals['Target'].pct_change() * 100
    
    furniture_goals['Month Name'] = furniture_goals['Month'].dt.strftime('%B')
    
    print("Furniture Category Monthly Target Changes:")
    print(furniture_goals[['Month Name', 'Target', 'MoM Change (%)']])
    
    notable_changes = furniture_goals[abs(furniture_goals['MoM Change (%)']) > 10].copy()
    
    print("\nMonths with Notable Target Fluctuations (>10%):")
    if len(notable_changes) > 0:
        print(notable_changes[['Month Name', 'Target', 'MoM Change (%)']])
    else:
        print("No months with fluctuations greater than 10% found.")
    
    # Visualizations
    # 1. Line chart of targets by month
    plt.figure(figsize=(14, 8))
    
    plt.plot(furniture_goals['Month Name'], furniture_goals['Target'], 
            marker='o', markersize=10, linewidth=2, color='blue')
    for x_val, y_val in zip(furniture_goals['Month Name'], furniture_goals['Target']):
        plt.annotate(f'${y_val:,.0f}', (x_val, y_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.title('Monthly Sales Targets for Furniture Category', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Target Amount', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/furniture_monthly_targets.png', dpi=300)
    
    # 2. Bar chart showing month to month % changes
    plt.figure(figsize=(14, 8))

    bar_colors = ['red' if x < -10 else 'green' if x > 10 else 'gray' 
             for x in furniture_goals['MoM Change (%)'].fillna(0)]

    month_names = furniture_goals['Month Name'][1:]
    percent_changes = furniture_goals['MoM Change (%)'][1:]
    
    bar_plot = plt.bar(month_names, percent_changes, color=bar_colors[1:])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    for bar in bar_plot:
        height = bar.get_height()
        if np.isnan(height):
            continue
        plt.text(bar.get_x() + bar.get_width()/2.,
                height + (5 if height > 0 else -10),
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9)

    plt.title('Month-over-Month % Change in Furniture Category Targets', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Percentage Change (%)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/furniture_target_changes.png', dpi=300)
    
    # 3. Heat map for month to month changes
    monthly_data = furniture_goals.copy()
    monthly_data['Change Category'] = pd.cut(
        monthly_data['MoM Change (%)'],
        bins=[-float('inf'), -15, -5, 5, 15, float('inf')],
        labels=['Large Decrease', 'Small Decrease', 'Stable', 'Small Increase', 'Large Increase']
    )

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    months_ordered = ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['Month Order'] = monthly_data['Month Name'].apply(lambda x: months_ordered.index(x))
    monthly_data = monthly_data.sort_values('Month Order')

    for i, (_, data_row) in enumerate(monthly_data.iterrows()):
        month_name = data_row['Month Name']
        percent_change = data_row['MoM Change (%)']
        target_val = data_row['Target']
        
        grid_row = 2 - (i // 4) 
        grid_col = i % 4

        if pd.isna(percent_change):
            cell_color = 'lightgray' 
        elif percent_change < -10:
            cell_color = 'tomato'
        elif percent_change < 0:
            cell_color = 'lightsalmon'
        elif percent_change == 0:
            cell_color = 'lightgray'
        elif percent_change < 10:
            cell_color = 'palegreen'
        else:
            cell_color = 'mediumseagreen'
        
        rectangle = plt.Rectangle((grid_col, grid_row), 0.9, 0.9, facecolor=cell_color, alpha=0.8, edgecolor='black')
        ax.add_patch(rectangle)

        ax.text(grid_col + 0.45, grid_row + 0.7, month_name, ha='center', va='center', fontweight='bold')

        ax.text(grid_col + 0.45, grid_row + 0.45, f'${target_val:,.0f}', ha='center', va='center')

        if not pd.isna(percent_change):
            ax.text(grid_col + 0.45, grid_row + 0.2, f'{percent_change:+.1f}%', ha='center', va='center', 
                   color='green' if percent_change > 0 else 'red')
    
    plt.title('Furniture Category Targets and Month-over-Month Changes', fontsize=16, fontweight='bold')
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='tomato', edgecolor='black', label='Large Decrease (< -10%)'),
        Patch(facecolor='lightsalmon', edgecolor='black', label='Small Decrease (0% to -10%)'),
        Patch(facecolor='lightgray', edgecolor='black', label='No Change (0%)'),
        Patch(facecolor='palegreen', edgecolor='black', label='Small Increase (0% to +10%)'),
        Patch(facecolor='mediumseagreen', edgecolor='black', label='Large Increase (> +10%)')
    ]
    ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig('results/furniture_target_calendar.png', dpi=300)
    
    return furniture_goals

# Part 3) Regional Performance Insights
def examine_regional_performance(combined_dataset):
    print("\n========== PART 3: REGIONAL PERFORMANCE INSIGHTS ==========\n")

    state_order_tally = combined_dataset.groupby('State')['Order ID'].nunique().reset_index()
    state_order_tally.columns = ['State', 'Order Count']
    state_order_tally = state_order_tally.sort_values('Order Count', ascending=False)

    top_five_states = state_order_tally.head(5)['State'].tolist()
    print(f"Top 5 states by order count: {', '.join(top_five_states)}")

    top_states_dataset = combined_dataset[combined_dataset['State'].isin(top_five_states)]

    state_metrics = top_states_dataset.groupby('State').agg({
        'Amount': 'sum',
        'Profit': ['sum', 'mean'],
        'Order ID': 'nunique'
    }).reset_index()

    state_metrics.columns = ['State', 'Total Sales', 'Total Profit', 'Average Profit', 'Order Count']
    state_metrics['Profit Margin (%)'] = (state_metrics['Total Profit'] / state_metrics['Total Sales'] * 100)
    state_metrics = state_metrics.sort_values('Total Sales', ascending=False)
    
    print("\nPerformance Metrics for Top 5 States:")
    print(state_metrics)
    highest_revenue_state = state_metrics.iloc[0]['State']
    lowest_revenue_state = state_metrics.iloc[-1]['State']
    
    best_margin_state = state_metrics.loc[state_metrics['Profit Margin (%)'].idxmax()]['State']
    worst_margin_state = state_metrics.loc[state_metrics['Profit Margin (%)'].idxmin()]['State']
    
    print(f"\nHighest revenue state: {highest_revenue_state}")
    print(f"Lowest revenue state: {lowest_revenue_state}")
    print(f"Best profit margin state: {best_margin_state}")
    print(f"Worst profit margin state: {worst_margin_state}")

    city_metrics = top_states_dataset.groupby(['State', 'City']).agg({
        'Amount': 'sum',
        'Profit': ['sum', 'mean'],
        'Order ID': 'nunique'
    }).reset_index()

    city_metrics.columns = ['State', 'City', 'Total Sales', 'Total Profit', 'Average Profit', 'Order Count']

    city_metrics['Profit Margin (%)'] = (city_metrics['Total Profit'] / city_metrics['Total Sales'] * 100)

    city_metrics = city_metrics.sort_values(['State', 'Total Sales'], ascending=[True, False])
    
    print("\nTop 2 Cities by Sales in Each of the Top 5 States:")
    for state in top_five_states:
        top_cities = city_metrics[city_metrics['State'] == state].head(2)
        print(f"\n{state}:")
        print(top_cities[['City', 'Total Sales', 'Total Profit', 'Profit Margin (%)', 'Order Count']])
    
    # Visualizations
    # 1. Bar chart of total sales by state
    plt.figure(figsize=(12, 7))
    axis = sns.barplot(x='State', y='Total Sales', data=state_metrics, palette='viridis')

    for i, p in enumerate(axis.patches):
        axis.annotate(f'${p.get_height():,.0f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.title('Total Sales by State (Top 5)', fontsize=16, fontweight='bold')
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Total Sales', fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/sales_by_state.png', dpi=300)
    
    # 2. Profit margin by state
    plt.figure(figsize=(12, 7))
    axis = sns.barplot(x='State', y='Profit Margin (%)', data=state_metrics, palette='coolwarm')

    for i, p in enumerate(axis.patches):
        axis.annotate(f'{p.get_height():.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=12)
    
    plt.title('Profit Margin by State (Top 5)', fontsize=16, fontweight='bold')
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Profit Margin (%)', fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/profit_margin_by_state.png', dpi=300)
    
    # 3. Sales and order count by state
    fig, primary_axis = plt.subplots(figsize=(12, 7))

    primary_axis.bar(state_metrics['State'], state_metrics['Total Sales'], color='skyblue', alpha=0.7)
    primary_axis.set_xlabel('State', fontsize=14)
    primary_axis.set_ylabel('Total Sales', fontsize=14, color='royalblue')
    primary_axis.tick_params(axis='y', labelcolor='royalblue')

    secondary_axis = primary_axis.twinx()
    secondary_axis.plot(state_metrics['State'], state_metrics['Order Count'], 'ro-', linewidth=2, markersize=8)
    secondary_axis.set_ylabel('Order Count', fontsize=14, color='red')
    secondary_axis.tick_params(axis='y', labelcolor='red')
    
    plt.title('Sales vs. Order Count by State (Top 5)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/sales_vs_orders_by_state.png', dpi=300)
    
    # 4. Top cities visualization
    leading_cities = pd.DataFrame()
    for state in top_five_states:
        leading_cities = pd.concat([leading_cities, city_metrics[city_metrics['State'] == state].head(2)])
    
    # Sorting by total sales
    leading_cities = leading_cities.sort_values('Total Sales', ascending=False).head(10)
    
    plt.figure(figsize=(14, 8))
    axis = sns.barplot(x='Total Sales', y='City', hue='State', data=leading_cities, palette='Set2')

    for i, p in enumerate(axis.patches):
        axis.annotate(f'${p.get_width():,.0f}', 
                   (p.get_width(), p.get_y() + p.get_height() / 2),
                   ha = 'left', va = 'center', fontsize=10)
    
    plt.title('Top 10 Cities by Sales', fontsize=16, fontweight='bold')
    plt.xlabel('Total Sales', fontsize=14)
    plt.ylabel('City', fontsize=14)
    plt.legend(title='State')
    plt.tight_layout()
    plt.savefig('results/top_cities_by_sales.png', dpi=300)
    
    # 5. Category performance by state
    state_product_groups = top_states_dataset.groupby(['State', 'Category']).agg({
        'Amount': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    state_product_groups['Profit Margin (%)'] = (state_product_groups['Profit'] / state_product_groups['Amount'] * 100)
    state_product_groups = state_product_groups.rename(columns={'Amount': 'Total Sales', 'Profit': 'Total Profit'})
    cross_table = state_product_groups.pivot_table(
        values='Total Sales', 
        index='State', 
        columns='Category'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cross_table, annot=True, cmap='YlGnBu', fmt=',.0f')
    plt.title('Category Sales by State (Top 5 States)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/category_sales_by_state.png', dpi=300)
    
    # 6. Bubble chart between sales, profit, and orders
    plt.figure(figsize=(12, 8))

    plt.scatter(
        state_metrics['Total Sales'], 
        state_metrics['Profit Margin (%)'],
        s=state_metrics['Order Count']*5,  
        c=range(len(state_metrics)),  
        cmap='viridis',
        alpha=0.7
    )

    for i, row in state_metrics.iterrows():
        plt.annotate(
            row['State'],
            (row['Total Sales'], row['Profit Margin (%)']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12
        )
    
    plt.title('Sales vs. Profit Margin by State', fontsize=16, fontweight='bold')
    plt.xlabel('Total Sales', fontsize=14)
    plt.ylabel('Profit Margin (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/state_sales_profit_bubble.png', dpi=300)
    
    return state_metrics, city_metrics

def compile_summary_report(product_group_summary, furniture_goals, state_metrics):
    print("\n========== SUMMARY REPORT ==========\n")
    
    # Part 1) Sales and Profitability Analysis Summary
    print("PART 1: SALES AND PROFITABILITY ANALYSIS")
    print("----------------------------------------")

    leading_category = product_group_summary.iloc[0]['Category']
    lagging_category = product_group_summary.iloc[-1]['Category']
    
    most_profitable_category = product_group_summary.loc[product_group_summary['Profit Margin (%)'].idxmax()]['Category']
    least_profitable_category = product_group_summary.loc[product_group_summary['Profit Margin (%)'].idxmin()]['Category']
    
    print(f"Leading category by sales: {leading_category} (${product_group_summary.iloc[0]['Total Sales']:,.2f})")
    print(f"Lagging category by sales: {lagging_category} (${product_group_summary.iloc[-1]['Total Sales']:,.2f})")
    print(f"Most profitable category: {most_profitable_category} ({product_group_summary.loc[product_group_summary['Profit Margin (%)'].idxmax()]['Profit Margin (%)']:.2f}%)")
    print(f"Least profitable category: {least_profitable_category} ({product_group_summary.loc[product_group_summary['Profit Margin (%)'].idxmin()]['Profit Margin (%)']:.2f}%)")

    print("\nKey Insights:")
    print(f"- {leading_category} generates the highest sales but may need optimization if profit margins are lower")
    print(f"- {most_profitable_category} provides the best profit margin and should be emphasized in marketing")
    print(f"- {least_profitable_category} may need cost reduction strategies or price adjustments")

    print("\nRecommendations:")
    print("1. Focus marketing efforts on high-margin categories while maintaining visibility of top sellers")
    print("2. Investigate cost structure of low-margin categories to identify optimization opportunities")
    print("3. Consider bundle strategies that combine high-margin items with high-volume sellers")
    print("4. Evaluate pricing strategies across categories to balance volume and profitability")
    
    # Part 2) Target Achievement Analysis Summary
    print("\nPART 2: TARGET ACHIEVEMENT ANALYSIS")
    print("-----------------------------------")

    significant_increases = furniture_goals[furniture_goals['MoM Change (%)'] > 10].copy()
    significant_decreases = furniture_goals[furniture_goals['MoM Change (%)'] < -10].copy()
    
    print("Key Findings:")
    
    if len(significant_increases) > 0:
        print(f"- Significant target increases occurred in: {', '.join(significant_increases['Month Name'])}")
    else:
        print("- No months with significant target increases (>10%) were found")
        
    if len(significant_decreases) > 0:
        print(f"- Significant target decreases occurred in: {', '.join(significant_decreases['Month Name'])}")
    else:
        print("- No months with significant target decreases (>10%) were found")

    peak_months = furniture_goals.nlargest(3, 'Target')['Month Name'].tolist()
    low_months = furniture_goals.nsmallest(3, 'Target')['Month Name'].tolist()
    
    print(f"- Highest target months: {', '.join(peak_months)}")
    print(f"- Lowest target months: {', '.join(low_months)}")

    print("\nRecommendations for Target Setting:")
    print("1. Establish more consistent month-to-month target changes to avoid drastic fluctuations")
    print("2. Align targets with seasonal patterns observed in historical data")
    print("3. Use a rolling forecast model that considers recent performance trends")
    print("4. Implement gradual increases rather than sudden jumps to make targets more achievable")
    print("5. Break down monthly targets into weekly milestones for better tracking and course correction")
    
    # Part 3) Regional Performance Insights Summary
    print("\nPART 3: REGIONAL PERFORMANCE INSIGHTS")
    print("-------------------------------------")

    best_sales_state = state_metrics.iloc[0]['State']
    worst_sales_state = state_metrics.iloc[-1]['State']
    
    best_margin_state = state_metrics.loc[state_metrics['Profit Margin (%)'].idxmax()]['State']
    worst_margin_state = state_metrics.loc[state_metrics['Profit Margin (%)'].idxmin()]['State']
    
    print(f"Best performing state by sales: {best_sales_state} (${state_metrics.iloc[0]['Total Sales']:,.2f})")
    print(f"Lowest performing state (of top 5): {worst_sales_state} (${state_metrics.iloc[-1]['Total Sales']:,.2f})")
    print(f"Best profit margin state: {best_margin_state} ({state_metrics.loc[state_metrics['Profit Margin (%)'].idxmax()]['Profit Margin (%)']:.2f}%)")
    print(f"Worst profit margin state: {worst_margin_state} ({state_metrics.loc[state_metrics['Profit Margin (%)'].idxmin()]['Profit Margin (%)']:.2f}%)")

    print("\nRegional Strategy Recommendations:")
    print(f"1. Leverage successful strategies from {best_sales_state} in other regions")
    print(f"2. Implement profit optimization strategies used in {best_margin_state} across other states")
    print(f"3. Focus improvement efforts on {worst_margin_state} to identify and address profitability issues")
    print("4. Develop region-specific product mixes based on category performance in each state")
    print("5. Consider geographical expansion based on proximity to high-performing regions")
    
    # Simple terms visual content
    plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2])
    ax1 = plt.subplot(gs[0, 0])
    product_group_summary.plot(
        kind='pie', 
        y='Total Sales', 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        ax=ax1,
        legend=False,
        colors=sns.color_palette('viridis', len(product_group_summary))
    )
    ax1.set_ylabel('')
    ax1.set_title('Sales Distribution by Category', fontsize=14)
    
    color_handles = [plt.Rectangle((0,0),1,1, color=sns.color_palette('viridis', len(product_group_summary))[i]) 
              for i in range(len(product_group_summary))]
    ax1.legend(color_handles, product_group_summary['Category'], loc='best', fontsize=10)
    
    ax2 = plt.subplot(gs[0, 1])
    sns.barplot(
        x='Profit Margin (%)', 
        y='Category', 
        data=product_group_summary.sort_values('Profit Margin (%)', ascending=False),
        palette='coolwarm',
        ax=ax2
    )
    ax2.set_title('Profit Margin by Category', fontsize=14)

    ax3 = plt.subplot(gs[1, 0])
    sns.barplot(
        x='State', 
        y='Total Sales', 
        data=state_metrics,
        palette='viridis',
        ax=ax3
    )
    ax3.set_title('Top 5 States by Sales', fontsize=14)
    ax3.set_ylabel('Total Sales')
    ax3.tick_params(axis='x', rotation=45)

    ax4 = plt.subplot(gs[1, 1])

    month_labels = furniture_goals['Month Name']
    target_values = furniture_goals['Target']
    
    ax4.plot(month_labels, target_values, marker='o', linestyle='-', color='blue', linewidth=2)
    ax4.set_title('Furniture Monthly Sales Targets', fontsize=14)
    ax4.set_ylabel('Target Amount')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Sales Analysis Summary Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/sales_analysis_summary.png', dpi=300)
    
    print("\nSummary visualization saved as 'results/sales_analysis_summary.png'")
    print("\nAnalysis complete. All visualizations saved to 'results/' directory.")

def main():
    purchase_items, revenue_goals, purchase_records = import_datasets()
    
    if purchase_items is None or revenue_goals is None or purchase_records is None:
        print("Error importing datasets. Exiting.")
        return
    
    combined_dataset, product_group_summary, subcategory_metrics = examine_sales_profit(purchase_items, purchase_records)
    furniture_goals = examine_target_achievement(revenue_goals)
    state_metrics, city_metrics = examine_regional_performance(combined_dataset)
    
    compile_summary_report(product_group_summary, furniture_goals, state_metrics)

if __name__ == "__main__":
    main()