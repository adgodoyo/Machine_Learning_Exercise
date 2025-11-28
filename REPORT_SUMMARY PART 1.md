# Part 1: Air Pollution Monitoring System - Report Summary

## Report Files Created

1. **Main Report:** `Part1_AirPollution_Report.md` (53 KB)
2. **Figures Folder:** `report_images/` (24 figures, ~2.3 MB total)

## Report Structure

The comprehensive report includes ALL required sections with detailed analysis:

### Section A: Missing Data Analysis
- Missing value percentages for each feature
- Missing data pattern visualization
- KNN imputation strategy with missing indicators
- Statistical validation of imputation quality

### Section B: Correlation Analysis
- Correlation heatmap showing multicollinearity
- VIF (Variance Inflation Factor) analysis
- Strong correlations identified (C6H6-NMHC: 0.98, CO-C6H6: 0.93)

### Section C: Ordinary Least Squares (OLS)
- Full statsmodels regression summary
- Significant variables identified (11 out of 12)
- Multicollinearity issues explained (Condition Number: 61,816)
- Performance: Test R² = 0.916, RMSE = 0.419

### Section D: Ridge & Lasso Regression
- Coefficient paths plotted vs. regularization parameter
- Ridge optimal α = 5.43
- Lasso optimal α = 0.00105
- Discussion of multicollinearity effects

### Section E: Kernel Ridge (Polynomial deg=2)
- Performance comparison with linear models
- Analysis of why polynomial kernel doesn't improve much (relationship already linear)

### Section F: Kernel Ridge (RBF)
- Grid search over λ and kernel width
- Best parameters identified
- Performance improvements noted

### Section G: Model Performance Comparison
- All 5 models compared across 6 metrics (Train/Test: R², MAE, RMSE)
- Best model: Kernel Ridge RBF (Test R² = 0.935, RMSE = 0.367)
- Interpretability vs performance trade-offs discussed

### Section H: Cost-Effective Sensor Selection
- **Cost vs RMSE frontier** plotted using Lasso path
- Low-cost design (≤£2,500): 4 sensors, RMSE = 0.44
- High-performance design (≤£4,000): 7 sensors, RMSE = 0.40
- Clear visualization of diminishing returns

### Section I: Alert Threshold Optimization
- Legal threshold: 5 mg/m³
- Cost structure: FP = £2,000, FN = £10,000
- Optimal threshold identified via cost curve
- Expected costs computed for different thresholds

### Section J: Monitor Design Comparison
- Expected costs: Low-cost vs High-performance
- Sensor failure robustness analysis
- **Recommendation:** High-performance justified for critical locations

### BONUS Section K: Classification Approach
- Logistic regression for binary classification (CO ≥ 5)
- Cost-sensitive threshold optimization (optimal ~0.35, not 0.5)
- Comparison with regression approach
- Practical deployment recommendations

## Key Findings

### 1. Multicollinearity is Severe
- Many sensors measure correlated pollutants (co-emitted from traffic)
- OLS coefficients unstable, regularization essential
- VIF values > 400 for some features

### 2. Feature Importance (from Lasso)
- **Most Important:** C6H6(GT), PT08.S1(CO), NOx(GT), NO2(GT)
- **Least Important:** Humidity variables, some redundant sensors

### 3. Cost-Accuracy Trade-off
- Diminishing returns after ~£2,500-£3,000
- **Low-cost monitor (£2,350):** 4 sensors, RMSE = 0.44
- **High-performance (£3,900):** 7 sensors, RMSE = 0.40
- Only 9% RMSE improvement for 66% cost increase

### 4. Alert System Economics
- Optimal alert threshold is **NOT** the legal limit (5 mg/m³)
- Must account for asymmetric costs (FN >> FP)
- Classification approach can reduce expected costs by 20-40%

### 5. Model Recommendations
- **For interpretability:** Lasso (sparse, clear feature selection)
- **For accuracy:** Kernel Ridge RBF (best test performance)
- **For alerts:** Logistic classification (better calibrated probabilities)

## Visualizations Included

All 24 figures are embedded in the report:
1. Missing data bar chart
2. Missing data pattern heatmap  
3-4. Missing data vs target analysis
5. Imputation quality comparison
6. Correlation heatmap (11x11)
7. Pairplot (key features)
8. VIF bar chart
9-10. OLS residual plots
11. Predicted vs actual (OLS)
12-13. Ridge coefficient paths
14. Lasso coefficient comparison
15. Polynomial kernel performance
16. RBF grid search heatmap
17-18. Model comparison charts
19. Cost vs RMSE frontier **KEY GRAPH**
20. Sensor selection visualization
21-22. Alert threshold cost curves **KEY GRAPHS**
23. Expected cost comparison
24. Sensor failure robustness

## Usage

To view the report:
```bash
# Open the markdown file
open Part1_AirPollution_Report.md

# Or convert to PDF
pandoc Part1_AirPollution_Report.md -o Part1_Report.pdf
```

All images are in `report_images/` folder and are referenced correctly in the markdown.

## Report Highlights

✓ All assignment questions (A-J) fully addressed
✓ 24 professional visualizations with captions
✓ Detailed statistical analysis and interpretation
✓ Model performance tables with 6 metrics each
✓ Cost-benefit analysis with recommendations
✓ Both English technical analysis
✓ Ready for submission after PDF conversion

**Total Length:** ~1,300 lines, ~53 KB
**Estimated Reading Time:** 30-40 minutes
