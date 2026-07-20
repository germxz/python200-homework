import os
import pandas as pd
from prefect import task, flow,  get_run_logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# task 1 load mutiple years of data 
@task(retries=3, retry_delay_seconds=2)
def process_happiness_data():
    logger = get_run_logger()

    base_url = (
        "https://raw.githubusercontent.com/"
        "Code-the-Dream-School/python-200-v1/"
        "main/assignments/resources/happiness_project"
    )

    all_dataframes = [] 

    for year in range(2015, 2025):
        file_path = f"{base_url}/world_happiness_{year}.csv"

        logger.info(f"Loading {file_path}")

        df = pd.read_csv(
            file_path,
            sep=";",
            decimal=","
        )

        df["Year"] = year
        all_dataframes.append(df)

    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_df["Happiness"] = merged_df["Happiness score"].fillna(
    merged_df["Ladder score"]
)
    os.makedirs("assignments_01/outputs", exist_ok=True)

    merged_df.to_csv(
        "assignments_01/outputs/merged_happiness.csv",
        index=False
    )

    logger.info("Merged dataset saved to assignments_01/outputs/merged_happiness.csv")

    return merged_df


# task 2 descriptive stats
@task
def descriptive_statistics(df):
    logger = get_run_logger()

    mean = df["Happiness"].mean()
    median = df["Happiness"].median()
    std = df["Happiness"].std()

    logger.info(f"Mean Happiness: {mean:.2f}")
    logger.info(f"Median Happiness: {median:.2f}")
    logger.info(f"Standard Deviation: {std:.2f}")

    year_means = df.groupby("Year")["Happiness"].mean()
    logger.info(f"\nAverage Happiness by Year:\n{year_means}")

    region_means = (
        df.groupby("Regional indicator")["Happiness"]
        .mean()
        .sort_values(ascending=False)
    )

    logger.info(f"\nAverage Happiness by Region:\n{region_means}")
    
    
#task 3 visual exploration
@task
def visual_exploration(df):
    logger = get_run_logger()
    
    # Plot 1: Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["Happiness"], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Happiness Scores Across All Years")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("assignments_01/outputs/happiness_histogram.png", dpi=300)
    logger.info("Histogram saved to assignments_01/outputs/happiness_histogram.png")
    plt.close()

    # Plot 2: Boxplot by year
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Year", y="Happiness")
    plt.title("Happiness Score Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.grid(True)
    plt.savefig("assignments_01/outputs/happiness_by_year.png", dpi=300)
    logger.info("Boxplot saved to assignments_01/outputs/happiness_by_year.png")
    plt.close()

    # Plot 3: Scatter plot (GDP vs Happiness)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="GDP per capita", y="Happiness", hue="Year", palette="viridis")
    plt.title("GDP Per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.grid(True)
    plt.savefig("assignments_01/outputs/gdp_vs_happiness.png", dpi=300)
    logger.info("Scatter plot saved to assignments_01/outputs/gdp_vs_happiness.png")
    plt.close()

    # Plot 4: Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of Numeric Variables")
    plt.tight_layout()
    plt.savefig("assignments_01/outputs/correlation_heatmap.png", dpi=300)
    logger.info("Correlation heatmap saved to assignments_01/outputs/correlation_heatmap.png")
    plt.close()
    
# task 4 hypothesis testing
@task
def hypothesis_testing(df):
    logger = get_run_logger()
    
    # ===== TEST 1: 2019 vs 2020 (Pandemic) =====
    data_2019 = df[df["Year"] == 2019]["Happiness"]
    data_2020 = df[df["Year"] == 2020]["Happiness"]
    
    t_stat, p_value = stats.ttest_ind(data_2019, data_2020)
    
    mean_2019 = data_2019.mean()
    mean_2020 = data_2020.mean()
    
    logger.info(f"2019 vs 2020 T-Test Results:")
    logger.info(f"  2019 Mean Happiness: {mean_2019:.4f}")
    logger.info(f"  2020 Mean Happiness: {mean_2020:.4f}")
    logger.info(f"  T-Statistic: {t_stat:.4f}")
    logger.info(f"  P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        logger.info(f"  Interpretation: Happiness significantly changed from 2019 to 2020 (p < 0.05). Mean increased by {mean_2020 - mean_2019:.4f}.")
    else:
        logger.info(f"  Interpretation: No significant change in happiness between 2019 and 2020 (p >= 0.05).")
    
    # ===== Test 2: North America/ANZ vs Sub-Saharan Africa) =====
    north_america = df[df["Regional indicator"] == "North America and ANZ"]["Happiness"]
    sub_saharan = df[df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness"]
    
    t_stat_2, p_value_2 = stats.ttest_ind(north_america, sub_saharan)
    
    mean_na = north_america.mean()
    mean_ss = sub_saharan.mean()
    
    logger.info(f"\nNorth America/ANZ vs Sub-Saharan Africa T-Test Results:")
    logger.info(f"  North America/ANZ Mean: {mean_na:.4f}")
    logger.info(f"  Sub-Saharan Africa Mean: {mean_ss:.4f}")
    logger.info(f"  T-Statistic: {t_stat_2:.4f}")
    logger.info(f"  P-Value: {p_value_2:.4f}")
    
    if p_value_2 < 0.05:
        logger.info(f"  Interpretation: Happiness differs significantly between regions (p < 0.05). North America/ANZ is {mean_na - mean_ss:.4f} points higher.")
    else:
        logger.info(f"  Interpretation: No significant difference between regions (p >= 0.05).")


@task
def correlation_analysis(df):
    logger = get_run_logger()
    
    # Get numeric columns (exclude Year, Ranking, etc.)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Remove non-explanatory columns
    exclude = ['Happiness', 'Year']
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    logger.info(f"Testing correlations for {len(numeric_cols)} variables:")
    
    correlations = []
    
    for col in numeric_cols:
        valid_data = df[[col, 'Happiness']].dropna()
        
        if len(valid_data) > 0:
            r, p_value = stats.pearsonr(valid_data[col], valid_data['Happiness'])
            correlations.append({'variable': col, 'r': r, 'p_value': p_value})
            logger.info(f"  {col}: r={r:.4f}, p={p_value:.4f}")
    
    # ===== Bonferroni Correction =====
    num_tests = len(correlations)
    adjusted_alpha = 0.05 / num_tests
    
    logger.info(f"\nBonferroni Correction:")
    logger.info(f"  Number of tests: {num_tests}")
    logger.info(f"  Adjusted alpha: {adjusted_alpha:.6f}")
    
    logger.info(f"\nSignificant at original alpha (0.05):")
    for corr in correlations:
        if corr['p_value'] < 0.05:
            logger.info(f"  {corr['variable']}: r={corr['r']:.4f}, p={corr['p_value']:.4f}")
    
    logger.info(f"\nSignificant after Bonferroni correction (alpha={adjusted_alpha:.6f}):")
    for corr in correlations:
        if corr['p_value'] < adjusted_alpha:
            logger.info(f"  {corr['variable']}: r={corr['r']:.4f}, p={corr['p_value']:.4f}")


@task 
def summary_report(df):
    logger = get_run_logger()
    
    # 1. Total countries and years
    logger.info("This dataset has a total of {} countries and {} years of data.".format(
        df['Country'].nunique(), 
        df['Year'].nunique()
    ))
    
    # 2. Top 3 and bottom 3 regions by mean happiness
    region_means = df.groupby("Regional indicator")["Happiness"].mean().sort_values(ascending=False)
    top_3_regions = region_means.head(3)
    bottom_3_regions = region_means.tail(3)
    
    logger.info("Top 3 regions by mean happiness: {}".format(", ".join([f"{r} ({s:.2f})" for r, s in top_3_regions.items()])))
    logger.info("Bottom 3 regions by mean happiness: {}".format(", ".join([f"{r} ({s:.2f})" for r, s in bottom_3_regions.items()])))
    
    # 3. 2019 vs 2020 t-test result
    data_2019 = df[df["Year"] == 2019]["Happiness"]
    data_2020 = df[df["Year"] == 2020]["Happiness"]
    t_stat, p_value = stats.ttest_ind(data_2019, data_2020)
    mean_2019 = data_2019.mean()
    mean_2020 = data_2020.mean()
    
    if p_value < 0.05:
        logger.info(f"Pre/Post-2020 Finding: Happiness significantly increased from 2019 ({mean_2019:.2f}) to 2020 ({mean_2020:.2f}), a difference of {mean_2020 - mean_2019:.2f} points (t={t_stat:.2f}, p={p_value:.4f}). This suggests global happiness actually rose at the start of the pandemic.")
    else:
        logger.info(f"Pre/Post-2020 Finding: No significant difference in happiness between 2019 ({mean_2019:.2f}) and 2020 ({mean_2020:.2f}) (p={p_value:.4f}).")
    
    # 4. Variable most strongly correlated (after Bonferroni)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    exclude = ['Happiness', 'Year']
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    correlations = []
    for col in numeric_cols:
        valid_data = df[[col, 'Happiness']].dropna()
        if len(valid_data) > 0:
            r, p_val = stats.pearsonr(valid_data[col], valid_data['Happiness'])
            correlations.append({'variable': col, 'r': r, 'p_value': p_val})
    
    num_tests = len(correlations)
    adjusted_alpha = 0.05 / num_tests
    
    # Find strongest that survives Bonferroni
    significant_corrs = [c for c in correlations if c['p_value'] < adjusted_alpha]
    if significant_corrs:
        strongest = max(significant_corrs, key=lambda x: abs(x['r']))
        logger.info(f"Strongest predictor of happiness (after Bonferroni correction): {strongest['variable']} with correlation r={strongest['r']:.4f} (p={strongest['p_value']:.4f})")
    else:
        logger.info("No variables remain significantly correlated with happiness after Bonferroni correction.")


@flow
def happiness_pipeline():
    df = process_happiness_data()
    descriptive_statistics(df)
    visual_exploration(df)
    hypothesis_testing(df)
    correlation_analysis(df)  
    summary_report(df)

if __name__ == "__main__":
    happiness_pipeline()


