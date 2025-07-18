import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
	page_title="Regression Coefficient Interpretation Guide",
	page_icon="📊",
	layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #f5f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 0.5rem solid #1E88E5;
    }
    .highlight {
        background-color: #ffff99;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .interpretation {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .equation {
        text-align: center;
        margin: 1rem 0;
    }
    .note {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 0.5rem solid #FFC107;
    }
    .tip {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 0.5rem solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>Comprehensive Guide to Interpreting Regression Coefficients</h1>",
			unsafe_allow_html=True)

# Introduction
st.markdown("""
This interactive guide explains how to interpret regression coefficients in various scenarios:
- Linear regression with continuous variables
- Log transformations (log-level, level-log, log-log models)
- Models with dummy variables
- Interaction terms (including log-dummy interactions)
- Polynomial terms (quadratic and cubic)

Each section includes formulas, visualizations, and practical interpretation examples.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
	"Go to Section:",
	["Introduction",
	 "Linear Regression Basics",
	 "Log Transformations",
	 "Dummy Variables",
	 "Log & Dummy Interactions",
	 "Polynomial Terms",
	 "Practical Examples",
	 "Interactive Playground"]
)


# Helper function to create example plots
def generate_example_data(model_type, n=100, seed=42):
	np.random.seed(seed)
	x = np.linspace(1, 10, n)

	if model_type == "linear":
		y = 2 + 3 * x + np.random.normal(0, 1, n)
		return x, y

	elif model_type == "log_level":
		y = 2 + 3 * np.log(x) + np.random.normal(0, 0.5, n)
		return x, y

	elif model_type == "level_log":
		y = np.exp(1 + 0.2 * x + np.random.normal(0, 0.1, n))
		return x, y

	elif model_type == "log_log":
		y = np.exp(1 + 0.7 * np.log(x) + np.random.normal(0, 0.1, n))
		return x, y

	elif model_type == "quadratic":
		y = 2 + 1.5 * x - 0.1 * x ** 2 + np.random.normal(0, 2, n)
		return x, y

	elif model_type == "cubic":
		y = 2 + 0.5 * x + 0.2 * x ** 2 - 0.03 * x ** 3 + np.random.normal(0, 2, n)
		return x, y

	elif model_type == "dummy":
		d = np.random.binomial(1, 0.5, n)
		y = 2 + 3 * x + 4 * d - 1.5 * x * d + np.random.normal(0, 1, n)
		return x, y, d


# Create coefficient interpretation plots
def create_interpretation_plot(model_type):
	if model_type == "linear":
		x, y = generate_example_data("linear")
		fig = px.scatter(x=x, y=y, opacity=0.7)

		# Add line of best fit
		b0, b1 = 2, 3  # Intercept and slope
		x_range = np.linspace(min(x), max(x), 100)
		y_fit = b0 + b1 * x_range

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit, mode='lines',
					   name='y = 2 + 3x',
					   line=dict(color='red', width=3))
		)

		# Highlight the interpretation of slope
		x_point = 5
		y_point = b0 + b1 * x_point

		fig.add_trace(
			go.Scatter(x=[x_point, x_point + 1],
					   y=[y_point, y_point + b1],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name='Δx=1 → Δy=3')
		)

		# Add annotations
		fig.add_annotation(
			x=x_point + 0.5,
			y=y_point + b1 / 2,
			text="Slope = 3",
			showarrow=True,
			arrowhead=2,
			ax=50,
			ay=-30
		)

		fig.update_layout(
			title="Linear Model: y = 2 + 3x",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig

	elif model_type == "log_level":
		x, y = generate_example_data("log_level")

		fig = px.scatter(x=x, y=y, opacity=0.7)

		# Add curve of best fit
		b0, b1 = 2, 3  # Intercept and coefficient
		x_range = np.linspace(min(x), max(x), 100)
		y_fit = b0 + b1 * np.log(x_range)

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit, mode='lines',
					   name='y = 2 + 3·ln(x)',
					   line=dict(color='red', width=3))
		)

		# Highlight the interpretation
		x_point = 5
		x_point_increase = x_point * 1.01  # 1% increase
		y_point = b0 + b1 * np.log(x_point)
		y_point_increase = b0 + b1 * np.log(x_point_increase)

		# Approximate change: b1 * 0.01
		approx_change = b1 * 0.01

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_increase],
					   y=[y_point, y_point_increase],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name='1% increase in x → Δy≈0.03')
		)

		fig.add_annotation(
			x=x_point + 0.025,
			y=y_point + approx_change / 2,
			text="Δy ≈ β·0.01 = 0.03",
			showarrow=True,
			arrowhead=2,
			ax=50,
			ay=-30
		)

		fig.update_layout(
			title="Log-Level Model: y = 2 + 3·ln(x)",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig

	elif model_type == "level_log":
		x, y = generate_example_data("level_log")

		fig = px.scatter(x=x, y=y, opacity=0.7)

		# Add curve of best fit
		b0, b1 = 1, 0.2  # Parameters in log scale
		x_range = np.linspace(min(x), max(x), 100)
		y_fit = np.exp(b0 + b1 * x_range)

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit, mode='lines',
					   name='ln(y) = 1 + 0.2x',
					   line=dict(color='red', width=3))
		)

		# Highlight the interpretation
		x_point = 5
		y_point = np.exp(b0 + b1 * x_point)
		x_point_increase = x_point + 1  # increase x by 1 unit
		y_point_increase = np.exp(b0 + b1 * x_point_increase)

		# Percentage change: (e^β - 1) * 100%
		percent_change = (np.exp(b1) - 1) * 100

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_increase],
					   y=[y_point, y_point_increase],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name='Δx=1 → %Δy≈22.1%')
		)

		fig.add_annotation(
			x=x_point + 0.5,
			y=(y_point + y_point_increase) / 2,
			text=f"Δx=1 → %Δy = {percent_change:.1f}%",
			showarrow=True,
			arrowhead=2,
			ax=50,
			ay=-30
		)

		fig.update_layout(
			title="Level-Log Model: ln(y) = 1 + 0.2x",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig

	elif model_type == "log_log":
		x, y = generate_example_data("log_log")

		fig = px.scatter(x=x, y=y, opacity=0.7)

		# Add curve of best fit
		b0, b1 = 1, 0.7  # Parameters in log scale
		x_range = np.linspace(min(x), max(x), 100)
		y_fit = np.exp(b0 + b1 * np.log(x_range))

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit, mode='lines',
					   name='ln(y) = 1 + 0.7·ln(x)',
					   line=dict(color='red', width=3))
		)

		# Highlight the interpretation
		x_point = 5
		y_point = np.exp(b0 + b1 * np.log(x_point))
		x_point_increase = x_point * 1.1  # increase x by 10%
		y_point_increase = np.exp(b0 + b1 * np.log(x_point_increase))

		# Percentage change: β * 10%
		percent_change = b1 * 10

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_increase],
					   y=[y_point, y_point_increase],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name='10% increase in x → %Δy≈7%')
		)

		fig.add_annotation(
			x=(x_point + x_point_increase) / 2,
			y=(y_point + y_point_increase) / 2,
			text=f"%Δx=10% → %Δy = {percent_change:.1f}%",
			showarrow=True,
			arrowhead=2,
			ax=50,
			ay=-30
		)

		fig.update_layout(
			title="Log-Log Model: ln(y) = 1 + 0.7·ln(x)",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig

	elif model_type == "quadratic":
		x, y = generate_example_data("quadratic")

		fig = px.scatter(x=x, y=y, opacity=0.7)

		# Add curve of best fit
		b0, b1, b2 = 2, 1.5, -0.1  # Parameters
		x_range = np.linspace(min(x), max(x), 100)
		y_fit = b0 + b1 * x_range + b2 * x_range ** 2

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit, mode='lines',
					   name='y = 2 + 1.5x - 0.1x²',
					   line=dict(color='red', width=3))
		)

		# Highlight the marginal effect at different points
		x_points = [2, 5, 8]

		for x_point in x_points:
			# Marginal effect: β₁ + 2β₂x
			marg_effect = b1 + 2 * b2 * x_point
			y_point = b0 + b1 * x_point + b2 * x_point ** 2

			fig.add_trace(
				go.Scatter(x=[x_point - 0.5, x_point + 0.5],
						   y=[y_point - marg_effect * 0.5, y_point + marg_effect * 0.5],
						   mode='lines',
						   line=dict(width=3, color='green'),
						   name=f'Slope at x={x_point}: {marg_effect:.2f}')
			)

			fig.add_annotation(
				x=x_point,
				y=y_point,
				text=f"Marginal effect: {marg_effect:.2f}",
				showarrow=True,
				arrowhead=2,
				ax=50,
				ay=30 if x_point != 5 else -30
			)

		# Add annotation for maximum point
		max_x = -b1 / (2 * b2)
		max_y = b0 + b1 * max_x + b2 * max_x ** 2

		fig.add_trace(
			go.Scatter(x=[max_x], y=[max_y],
					   mode='markers',
					   marker=dict(size=12, color='blue', symbol='star'),
					   name=f'Maximum at x={max_x:.1f}')
		)

		fig.add_annotation(
			x=max_x,
			y=max_y + 1,
			text=f"Maximum at x={max_x:.1f}",
			showarrow=True,
			arrowhead=2,
			ax=0,
			ay=-30
		)

		fig.update_layout(
			title="Quadratic Model: y = 2 + 1.5x - 0.1x²",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig

	elif model_type == "cubic":
		x, y = generate_example_data("cubic")

		fig = px.scatter(x=x, y=y, opacity=0.7)

		# Add curve of best fit
		b0, b1, b2, b3 = 2, 0.5, 0.2, -0.03  # Parameters
		x_range = np.linspace(min(x), max(x), 100)
		y_fit = b0 + b1 * x_range + b2 * x_range ** 2 + b3 * x_range ** 3

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit, mode='lines',
					   name='y = 2 + 0.5x + 0.2x² - 0.03x³',
					   line=dict(color='red', width=3))
		)

		# Highlight the marginal effect at different points
		x_points = [2, 5, 8]

		for x_point in x_points:
			# Marginal effect: β₁ + 2β₂x + 3β₃x²
			marg_effect = b1 + 2 * b2 * x_point + 3 * b3 * x_point ** 2
			y_point = b0 + b1 * x_point + b2 * x_point ** 2 + b3 * x_point ** 3

			fig.add_trace(
				go.Scatter(x=[x_point - 0.5, x_point + 0.5],
						   y=[y_point - marg_effect * 0.5, y_point + marg_effect * 0.5],
						   mode='lines',
						   line=dict(width=3, color='green'),
						   name=f'Slope at x={x_point}: {marg_effect:.2f}')
			)

			fig.add_annotation(
				x=x_point,
				y=y_point,
				text=f"Marginal effect: {marg_effect:.2f}",
				showarrow=True,
				arrowhead=2,
				ax=50,
				ay=30 if x_point != 5 else -30
			)

		fig.update_layout(
			title="Cubic Model: y = 2 + 0.5x + 0.2x² - 0.03x³",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig

	elif model_type == "dummy":
		x, y, d = generate_example_data("dummy")

		df = pd.DataFrame({
			'x': x,
			'y': y,
			'dummy': d
		})

		fig = px.scatter(df, x='x', y='y', color='dummy', opacity=0.7,
						 color_discrete_sequence=['blue', 'red'])

		# Add lines of best fit
		b0, b1, b2, b3 = 2, 3, 4, -1.5  # Parameters
		x_range = np.linspace(min(x), max(x), 100)

		# For dummy = 0
		y_fit_0 = b0 + b1 * x_range

		# For dummy = 1
		y_fit_1 = (b0 + b2) + (b1 + b3) * x_range

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit_0, mode='lines',
					   name='dummy=0: y = 2 + 3x',
					   line=dict(color='blue', width=3))
		)

		fig.add_trace(
			go.Scatter(x=x_range, y=y_fit_1, mode='lines',
					   name='dummy=1: y = (2+4) + (3-1.5)x',
					   line=dict(color='red', width=3))
		)

		# Add annotations
		fig.add_annotation(
			x=8,
			y=b0 + b1 * 8,
			text="Slope (dummy=0): 3",
			showarrow=True,
			arrowhead=2,
			ax=-50,
			ay=-30
		)

		fig.add_annotation(
			x=8,
			y=(b0 + b2) + (b1 + b3) * 8,
			text="Slope (dummy=1): 1.5",
			showarrow=True,
			arrowhead=2,
			ax=50,
			ay=30
		)

		fig.add_annotation(
			x=1,
			y=(b0 + b2 + (b1 + b3) * 1 + b0 + b1 * 1) / 2,
			text="Intercept difference: 4",
			showarrow=True,
			arrowhead=2,
			ax=0,
			ay=-50
		)

		fig.update_layout(
			title="Model with Dummy Variable and Interaction: y = 2 + 3x + 4·dummy - 1.5x·dummy",
			xaxis_title="x",
			yaxis_title="y",
			height=500
		)

		return fig


# Introduction section
if section == "Introduction":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Understanding Regression Coefficients</h2>", unsafe_allow_html=True)

	st.markdown("""
    Interpreting regression coefficients correctly is essential for making meaningful inferences from your models. The interpretation varies depending on:

    1. Whether variables are in their natural units or transformed (e.g., logarithms)
    2. Whether variables are continuous or categorical (dummy variables)
    3. Whether the model includes interaction terms or polynomial terms

    This guide provides a comprehensive overview of how to interpret coefficients in various regression scenarios, with visual examples and practical applications.
    """)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Tips for interpreting coefficients:**

    * Always consider the units of your variables
    * Pay attention to model transformations (logs, squares, etc.)
    * Remember that interpretation is about the relationship between variables, holding other factors constant
    * Coefficients show correlation, not necessarily causation
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    In the following sections, we'll explore different regression models and how to interpret their coefficients correctly.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Show a preview of all model types
	st.markdown("<h2 class='sub-header'>Model Types Overview</h2>", unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("<div class='section'>", unsafe_allow_html=True)
		st.markdown("### Linear Model (Level-Level)")
		st.latex(r"Y = \beta_0 + \beta_1 X + \varepsilon")
		st.markdown("**Interpretation:** A one-unit change in X is associated with a $\\beta_1$ unit change in Y.")
		st.markdown("</div>", unsafe_allow_html=True)

		st.markdown("<div class='section'>", unsafe_allow_html=True)
		st.markdown("### Log-Level Model")
		st.latex(r"Y = \beta_0 + \beta_1 \ln(X) + \varepsilon")
		st.markdown("**Interpretation:** A 1% change in X is associated with a $\\beta_1/100$ unit change in Y.")
		st.markdown("</div>", unsafe_allow_html=True)

		st.markdown("<div class='section'>", unsafe_allow_html=True)
		st.markdown("### Models with Dummy Variables")
		st.latex(r"Y = \beta_0 + \beta_1 X + \beta_2 D + \varepsilon")
		st.markdown(
			"**Interpretation:** $\\beta_2$ represents the difference in Y between the group where D=1 and the group where D=0.")
		st.markdown("</div>", unsafe_allow_html=True)

	with col2:
		st.markdown("<div class='section'>", unsafe_allow_html=True)
		st.markdown("### Level-Log Model")
		st.latex(r"\ln(Y) = \beta_0 + \beta_1 X + \varepsilon")
		st.markdown(
			"**Interpretation:** A one-unit change in X is associated with a $(e^{\\beta_1} - 1) \\times 100\\%$ change in Y.")
		st.markdown("</div>", unsafe_allow_html=True)

		st.markdown("<div class='section'>", unsafe_allow_html=True)
		st.markdown("### Log-Log Model")
		st.latex(r"\ln(Y) = \beta_0 + \beta_1 \ln(X) + \varepsilon")
		st.markdown("**Interpretation:** A 1% change in X is associated with a $\\beta_1\\%$ change in Y (elasticity).")
		st.markdown("</div>", unsafe_allow_html=True)

		st.markdown("<div class='section'>", unsafe_allow_html=True)
		st.markdown("### Polynomial Models")
		st.latex(r"Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \varepsilon")
		st.markdown(
			"**Interpretation:** The marginal effect of X on Y is $\\beta_1 + 2\\beta_2 X + 3\\beta_3 X^2$, which varies with X.")
		st.markdown("</div>", unsafe_allow_html=True)

# Linear Regression Basics
elif section == "Linear Regression Basics":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Linear Regression: The Level-Level Model</h2>", unsafe_allow_html=True)

	st.markdown("""
    The standard linear regression model (also called the "level-level" model) is expressed as:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 X + \varepsilon")

	st.markdown("""
    where:
    - $Y$ is the dependent variable
    - $X$ is the independent variable
    - $\\beta_0$ is the intercept
    - $\\beta_1$ is the slope coefficient
    - $\\varepsilon$ is the error term

    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Intercept ($\\beta_0$)**: The expected value of $Y$ when $X = 0$.

    **Slope ($\\beta_1$)**: A one-unit increase in $X$ is associated with a $\\beta_1$ unit change in $Y$, holding all else constant.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model where salary ($Y$) is regressed on years of experience ($X$):

    $Salary = 30000 + 5000 \\times Experience$

    - The intercept ($30000) indicates the expected salary for someone with zero years of experience.
    - The slope ($5000) indicates that each additional year of experience is associated with a $5000 increase in salary, on average.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("linear")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** In a multiple regression model: $Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\beta_k X_k + \\varepsilon$

    Each coefficient $\\beta_j$ represents the expected change in $Y$ associated with a one-unit increase in $X_j$, holding all other variables constant.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Multiple variables section
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Multiple Regression Interpretation</h2>", unsafe_allow_html=True)

	st.markdown("""
    In multiple regression, we extend the simple model to include more than one predictor:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_k X_k + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients in Multiple Regression
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Coefficient $\\beta_j$**: A one-unit increase in $X_j$ is associated with a $\\beta_j$ unit change in $Y$, **holding all other variables constant**.

    The "holding all else constant" part is crucial - it means that each coefficient shows the isolated relationship between that predictor and the outcome when controlling for all other variables in the model.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting house price based on size and age:

    $Price = 100000 + 150 \\times Size - 2000 \\times Age$

    - A one square foot increase in size is associated with a $150 increase in price, holding age constant.
    - A one year increase in age is associated with a $2000 decrease in price, holding size constant.
    """)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Practical Tips:**

    1. Always report units when interpreting coefficients
    2. Be careful about extrapolating far beyond your data range
    3. Check for multicollinearity which can distort coefficient interpretations
    4. Remember that statistical significance doesn't equal practical significance
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

# Log Transformations
elif section == "Log Transformations":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Log Transformations in Regression</h2>", unsafe_allow_html=True)

	st.markdown("""
    Log transformations are commonly used in regression to:
    - Deal with skewed distributions
    - Model percentage changes and elasticities
    - Account for non-linear relationships
    - Stabilize variance

    There are three main types of log transformation models:
    1. Log-Level Model (only Y is logged)
    2. Level-Log Model (only X is logged)
    3. Log-Log Model (both Y and X are logged)
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Log-Level Model
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>1. Log-Level Model</h3>", unsafe_allow_html=True)

	st.markdown("""
    In a log-level model, we take the natural logarithm of the dependent variable:
    """)

	st.latex(r"\ln(Y) = \beta_0 + \beta_1 X + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Coefficient $\\beta_1$**: A one-unit increase in $X$ is associated with a **$(e^{\\beta_1} - 1) \\times 100\\%$ change** in $Y$, holding all else constant.

    For small values of $\\beta_1$ (roughly $|\\beta_1| < 0.1$), we can use the approximation: $\\beta_1 \\times 100\\%$
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model where $\\ln(Salary)$ is regressed on years of experience:

    $\\ln(Salary) = 10.5 + 0.08 \\times Experience$

    - A one-year increase in experience is associated with approximately an $(e^{0.08} - 1) \\times 100\\% = 8.33\\%$ increase in salary.
    - Using the approximation: $0.08 \\times 100\\% = 8\\%$ increase in salary.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("level_log")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** The log-level model is especially useful when:
    - The percentage change in Y for a unit change in X is constant
    - Y follows a multiplicative relationship with X
    - The effect of X on Y grows exponentially
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Level-Log Model
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>2. Level-Log Model</h3>", unsafe_allow_html=True)

	st.markdown("""
    In a level-log model, we take the natural logarithm of the independent variable:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 \ln(X) + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Coefficient $\\beta_1$**: A 1% increase in $X$ is associated with a $\\frac{\\beta_1}{100}$ unit change in $Y$, holding all else constant.

    Alternatively: A 100% increase (doubling) of $X$ is associated with a $\\beta_1 \\ln(2) \\approx 0.693\\beta_1$ unit change in $Y$.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model where test score is regressed on the logarithm of study hours:

    $Score = 60 + 10 \\times \\ln(StudyHours)$

    - A 1% increase in study hours is associated with a $\\frac{10}{100} = 0.1$ point increase in test score.
    - Doubling study hours is associated with a $10 \\times \\ln(2) \\approx 6.93$ point increase in test score.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("log_level")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** The level-log model is especially useful when:
    - The absolute change in Y for a percentage change in X is constant
    - The effect of X on Y diminishes as X increases (diminishing returns)
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Log-Log Model
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>3. Log-Log Model</h3>", unsafe_allow_html=True)

	st.markdown("""
    In a log-log model, we take the natural logarithm of both the dependent and independent variables:
    """)

	st.latex(r"\ln(Y) = \beta_0 + \beta_1 \ln(X) + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Coefficient $\\beta_1$**: A 1% increase in $X$ is associated with a $\\beta_1\\%$ change in $Y$, holding all else constant.

    This coefficient represents the **elasticity** of $Y$ with respect to $X$ - how responsive $Y$ is to changes in $X$ in percentage terms.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model where the log of income is regressed on the log of education years:

    $\\ln(Income) = 8.2 + 0.7 \\times \\ln(Education)$

    - A 1% increase in years of education is associated with a 0.7% increase in income.
    - If education increases by 10%, income is expected to increase by approximately 7%.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("log_log")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** The log-log model is especially useful for:
    - Analyzing elasticities (e.g., price elasticity of demand)
    - Modeling relationships that follow power laws
    - Data with wide ranges of values for both X and Y
    - When percentage changes are more meaningful than absolute changes
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Summary table for log transformations
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Summary of Log Transformations</h3>", unsafe_allow_html=True)

	summary_data = {
		"Model Type": ["Level-Level", "Log-Level", "Level-Log", "Log-Log"],
		"Equation": ["Y = β₀ + β₁X", "ln(Y) = β₀ + β₁X", "Y = β₀ + β₁ln(X)", "ln(Y) = β₀ + β₁ln(X)"],
		"Interpretation of β₁": ["One-unit increase in X → β₁ unit change in Y",
								 "One-unit increase in X → (e^β₁-1)×100% change in Y",
								 "1% increase in X → β₁/100 unit change in Y",
								 "1% increase in X → β₁% change in Y"],
		"Common Uses": ["Linear relationships",
						"Percentage changes in Y; when Y is skewed",
						"Diminishing returns; when X is skewed",
						"Elasticities; when both X and Y are skewed"]
	}

	summary_df = pd.DataFrame(summary_data)
	st.table(summary_df)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Practical Tips:**

    1. Use log transformations when data is positively skewed or spans several orders of magnitude
    2. Remember that log transformations only work with positive values
    3. For small coefficient values (< 0.1), you can interpret directly as percentage changes
    4. When reporting results, clearly state that you're using log-transformed variables and explain the interpretation
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

# Dummy Variables
elif section == "Dummy Variables":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Dummy Variables in Regression</h2>", unsafe_allow_html=True)

	st.markdown("""
    Dummy variables (also called indicator or binary variables) take the value of 0 or 1 to indicate the absence or presence of a categorical effect. They allow us to include categorical variables in regression models.
    """)

	st.markdown("""
    ### Basic Model with a Dummy Variable

    The simplest model with a dummy variable is:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 D + \varepsilon")

	st.markdown("""
    where $D$ is a dummy variable equal to 1 for observations in a certain category and 0 otherwise.

    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Intercept ($\\beta_0$)**: The expected value of $Y$ when $D = 0$ (the reference/base category).

    **Coefficient of dummy variable ($\\beta_1$)**: The difference in the expected value of $Y$ between the category where $D = 1$ and the category where $D = 0$.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting salary based on gender (male = 0, female = 1):

    $Salary = 50000 - 5000 \\times Female$

    - The expected salary for males (Female = 0) is $50,000.
    - The expected salary for females (Female = 1) is $45,000.
    - The coefficient -5000 represents the gender pay gap in this model.
    """)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** When interpreting dummy variables, it's important to understand which category is coded as 0 (the reference category) and which is coded as 1.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Model with continuous and dummy variables
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Model with Continuous and Dummy Variables</h3>", unsafe_allow_html=True)

	st.markdown("""
    When we include both continuous and dummy variables in a model:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 X + \beta_2 D + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Intercept ($\\beta_0$)**: The expected value of $Y$ when $X = 0$ and $D = 0$.

    **Coefficient of continuous variable ($\\beta_1$)**: The change in $Y$ associated with a one-unit increase in $X$, holding $D$ constant.

    **Coefficient of dummy variable ($\\beta_2$)**: The difference in the expected value of $Y$ between the category where $D = 1$ and the category where $D = 0$, holding $X$ constant.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting house price based on size and location (urban = 0, suburban = 1):

    $Price = 100000 + 200 \\times Size + 50000 \\times Suburban$

    - For urban houses (Suburban = 0), the expected price is $100,000 + $200 × Size.
    - For suburban houses (Suburban = 1), the expected price is $150,000 + $200 × Size.
    - The coefficient 50000 means that suburban houses are, on average, $50,000 more expensive than urban houses of the same size.
    """)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Geometric Interpretation:** The dummy variable shifts the intercept of the regression line, but the slope (effect of the continuous variable) remains the same for both categories.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Multiple categories
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Multiple Categories (One-Hot Encoding)</h3>", unsafe_allow_html=True)

	st.markdown("""
    For a categorical variable with $k$ categories, we use $k-1$ dummy variables to avoid the dummy variable trap (perfect multicollinearity):
    """)

	st.latex(r"Y = \beta_0 + \beta_1 D_1 + \beta_2 D_2 + ... + \beta_{k-1} D_{k-1} + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Intercept ($\\beta_0$)**: The expected value of $Y$ for the reference category (when all dummy variables = 0).

    **Coefficient of dummy variable ($\\beta_j$)**: The difference in the expected value of $Y$ between category $j$ and the reference category.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting wage based on education level (high school = reference, bachelor's, master's):

    $Wage = 40000 + 20000 \\times Bachelor + 35000 \\times Master$

    - The expected wage for high school graduates (reference category) is $40,000.
    - The expected wage for bachelor's degree holders is $60,000 ($40,000 + $20,000).
    - The expected wage for master's degree holders is $75,000 ($40,000 + $35,000).
    - The coefficient 20000 means bachelor's degree holders earn $20,000 more than high school graduates on average.
    - The coefficient 35000 means master's degree holders earn $35,000 more than high school graduates on average.
    """)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Practical Tips:**

    1. Always clearly identify the reference category when reporting results
    2. The choice of reference category affects the coefficients but not the overall model fit
    3. Consider choosing the most common category or a logical baseline as the reference
    4. Remember that you need k-1 dummies for k categories to avoid the dummy variable trap
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

# Log & Dummy Interactions
elif section == "Log & Dummy Interactions":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Interaction Terms in Regression</h2>", unsafe_allow_html=True)

	st.markdown("""
    Interaction terms allow the effect of one variable to depend on the value of another variable. This section covers:

    1. Interactions between continuous and dummy variables
    2. Interactions with log-transformed variables
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Continuous-Dummy Interactions
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Interactions Between Continuous and Dummy Variables</h3>",
				unsafe_allow_html=True)

	st.markdown("""
    A model with an interaction between a continuous variable $X$ and a dummy variable $D$:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 X + \beta_2 D + \beta_3 (X \times D) + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Intercept ($\\beta_0$)**: The expected value of $Y$ when $X = 0$ and $D = 0$.

    **Coefficient of continuous variable ($\\beta_1$)**: The effect of a one-unit increase in $X$ on $Y$ when $D = 0$.

    **Coefficient of dummy variable ($\\beta_2$)**: The difference in the expected value of $Y$ between the category where $D = 1$ and the category where $D = 0$ when $X = 0$.

    **Interaction coefficient ($\\beta_3$)**: The difference in the effect of $X$ on $Y$ between the category where $D = 1$ and the category where $D = 0$.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting salary based on experience and gender (male = 0, female = 1):

    $Salary = 30000 + 5000 \\times Experience + 2000 \\times Female - 1000 \\times (Experience \\times Female)$

    - For males (Female = 0): $Salary = 30000 + 5000 \\times Experience$
    - For females (Female = 1): $Salary = 32000 + 4000 \\times Experience$

    Interpretation:
    - The starting salary (Experience = 0) for males is $30,000.
    - The starting salary for females is $32,000 ($2,000 higher than males).
    - Each year of experience increases a male's salary by $5,000.
    - Each year of experience increases a female's salary by $4,000 (which is $1,000 less than for males).
    - The negative interaction coefficient (-1000) indicates that the return to experience is lower for females than for males.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("dummy")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Geometric Interpretation:** Without interaction, the regression lines for the two groups are parallel (same slope, different intercepts). With interaction, the lines have different slopes, indicating that the effect of the continuous variable differs between groups.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Log-Dummy Interactions
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Interactions with Log-Transformed Variables</h3>", unsafe_allow_html=True)

	st.markdown("""
    When we combine log transformations with dummy variables, the interpretation becomes more complex. Let's look at a few common scenarios:
    """)

	# Log-Level with Dummy
	st.markdown("""
    ### Log-Level Model with Dummy
    """)

	st.latex(r"\ln(Y) = \beta_0 + \beta_1 X + \beta_2 D + \beta_3 (X \times D) + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Coefficient of dummy variable ($\\beta_2$)**: When $X = 0$, being in the category where $D = 1$ is associated with a $(e^{\\beta_2} - 1) \\times 100\\%$ difference in $Y$ compared to the category where $D = 0$.

    **Coefficient of continuous variable ($\\beta_1$)**: When $D = 0$, a one-unit increase in $X$ is associated with a $(e^{\\beta_1} - 1) \\times 100\\%$ change in $Y$.

    **Interaction coefficient ($\\beta_3$)**: The difference in the percentage effect of $X$ on $Y$ between the two categories. When $D = 1$, a one-unit increase in $X$ is associated with a $(e^{\\beta_1 + \\beta_3} - 1) \\times 100\\%$ change in $Y$.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting wage (logged) based on education years and gender:

    $\\ln(Wage) = 2.5 + 0.1 \\times Education + 0.2 \\times Female - 0.03 \\times (Education \\times Female)$

    - For males (Female = 0): $\\ln(Wage) = 2.5 + 0.1 \\times Education$
    - For females (Female = 1): $\\ln(Wage) = 2.7 + 0.07 \\times Education$

    Interpretation:
    - With no education, females earn approximately $(e^{0.2} - 1) \\times 100\\% = 22.1\\%$ more than males.
    - For males, each additional year of education is associated with approximately $(e^{0.1} - 1) \\times 100\\% = 10.5\\%$ higher wages.
    - For females, each additional year of education is associated with approximately $(e^{0.1-0.03} - 1) \\times 100\\% = 7.3\\%$ higher wages.
    - The negative interaction coefficient (-0.03) indicates that the return to education is lower for females than for males in percentage terms.
    """)

	# Log-Log with Dummy
	st.markdown("""
    ### Log-Log Model with Dummy
    """)

	st.latex(r"\ln(Y) = \beta_0 + \beta_1 \ln(X) + \beta_2 D + \beta_3 (D \times \ln(X)) + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Coefficient of dummy variable ($\\beta_2$)**: When $X = 1$ (since $\\ln(1) = 0$), being in the category where $D = 1$ is associated with a $(e^{\\beta_2} - 1) \\times 100\\%$ difference in $Y$.

    **Coefficient of log variable ($\\beta_1$)**: When $D = 0$, a 1% increase in $X$ is associated with a $\\beta_1\\%$ change in $Y$.

    **Interaction coefficient ($\\beta_3$)**: The difference in elasticity between the two categories. When $D = 1$, a 1% increase in $X$ is associated with a $(\\beta_1 + \\beta_3)\\%$ change in $Y$.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting house price (logged) based on square footage (logged) and location (urban vs. suburban):

    $\\ln(Price) = 10 + 0.8 \\times \\ln(SqFt) + 0.3 \\times Suburban + 0.2 \\times (Suburban \\times \\ln(SqFt))$

    - For urban houses (Suburban = 0): $\\ln(Price) = 10 + 0.8 \\times \\ln(SqFt)$
    - For suburban houses (Suburban = 1): $\\ln(Price) = 10.3 + 1.0 \\times \\ln(SqFt)$

    Interpretation:
    - For a 1 sq.ft. house (theoretical), suburban houses are approximately $(e^{0.3} - 1) \\times 100\\% = 35\\%$ more expensive than urban houses.
    - For urban houses, a 1% increase in square footage is associated with a 0.8% increase in price (elasticity = 0.8).
    - For suburban houses, a 1% increase in square footage is associated with a 1.0% increase in price (elasticity = 1.0).
    - The positive interaction coefficient (0.2) indicates that the price elasticity with respect to size is higher for suburban houses than for urban houses.
    """)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Practical Tips for Interaction Terms:**

    1. Always include the main effects (individual variables) in a model with interaction terms
    2. Center continuous variables before creating interactions to make the main effects more interpretable
    3. Plot interaction effects to visualize how the relationship varies across groups
    4. Use marginal effects plots to show how the effect of one variable changes across the range of another variable
    5. For complex interactions, consider calculating the expected values at meaningful values of the variables
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

# Polynomial Terms
elif section == "Polynomial Terms":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Polynomial Terms in Regression</h2>", unsafe_allow_html=True)

	st.markdown("""
    Polynomial terms (squared, cubed, etc.) allow us to model non-linear relationships between variables. This section covers:

    1. Quadratic Models
    2. Cubic Models
    3. Interpreting marginal effects in polynomial models
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Quadratic Models
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Quadratic Models</h3>", unsafe_allow_html=True)

	st.markdown("""
    A quadratic model includes a squared term of a continuous variable:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    In a quadratic model, the effect of $X$ on $Y$ depends on the value of $X$ itself. 

    The **marginal effect** of $X$ on $Y$ is:
    """)

	st.latex(r"\frac{\partial Y}{\partial X} = \beta_1 + 2\beta_2 X")

	st.markdown("""
    This means that the effect of a one-unit increase in $X$ on $Y$ varies depending on the value of $X$.

    - If $\\beta_2 > 0$: The relationship is U-shaped (convex)
    - If $\\beta_2 < 0$: The relationship is inverted U-shaped (concave)

    The turning point (maximum or minimum) occurs at $X = -\\frac{\\beta_1}{2\\beta_2}$
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting wage based on years of experience and its square:

    $Wage = 20000 + 5000 \\times Experience - 100 \\times Experience^2$

    Marginal effect: $\\frac{\\partial Wage}{\\partial Experience} = 5000 - 200 \\times Experience$

    Interpretation:
    - The marginal effect of an additional year of experience depends on how much experience one already has.
    - At Experience = 0, an additional year increases wage by $5,000.
    - At Experience = 10, an additional year increases wage by $5,000 - 200 \\times 10 = $3,000.
    - At Experience = 25, an additional year increases wage by $5,000 - 200 \\times 25 = $0.
    - Beyond 25 years, additional experience is associated with a decrease in wage.
    - The turning point (maximum wage) occurs at Experience = $-\\frac{5000}{2 \\times (-100)} = 25$ years.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("quadratic")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** In a quadratic model, the individual coefficients ($\\beta_1$ and $\\beta_2$) don't have a straightforward interpretation on their own. It's more meaningful to interpret the combined marginal effect.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Cubic Models
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Cubic Models</h3>", unsafe_allow_html=True)

	st.markdown("""
    A cubic model includes both squared and cubed terms of a continuous variable:
    """)

	st.latex(r"Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \varepsilon")

	st.markdown("""
    ### Interpretation of Coefficients
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    Similar to quadratic models, the effect of $X$ on $Y$ depends on the value of $X$. 

    The **marginal effect** of $X$ on $Y$ is:
    """)

	st.latex(r"\frac{\partial Y}{\partial X} = \beta_1 + 2\beta_2 X + 3\beta_3 X^2")

	st.markdown("""
    This is a quadratic function of $X$, which means the marginal effect can change direction twice, allowing for more complex non-linear relationships.

    The cubic term ($\\beta_3$) determines the overall shape at large values of $X$:
    - If $\\beta_3 > 0$: $Y$ eventually increases as $X$ gets large
    - If $\\beta_3 < 0$: $Y$ eventually decreases as $X$ gets large
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("""
    ### Example

    In a model predicting crop yield based on fertilizer amount:

    $Yield = 10 + 5 \\times Fertilizer - 0.5 \\times Fertilizer^2 + 0.01 \\times Fertilizer^3$

    Marginal effect: $\\frac{\\partial Yield}{\\partial Fertilizer} = 5 - 1 \\times Fertilizer + 0.03 \\times Fertilizer^2$

    Interpretation:
    - At low fertilizer levels, adding more fertilizer increases yield.
    - As fertilizer increases further, its effectiveness diminishes (diminishing returns).
    - At very high fertilizer levels, the yield starts increasing again (though this might be beyond the range of realistic values).
    - The equation $5 - 1 \\times Fertilizer + 0.03 \\times Fertilizer^2 = 0$ gives the points where the marginal effect is zero.
    """)

	# Display the visualization
	st.markdown("### Visual Interpretation")
	fig = create_interpretation_plot("cubic")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** Cubic and higher-order polynomial models provide greater flexibility but can be harder to interpret and may lead to overfitting. Consider whether the added complexity is justified by theory or improves model fit significantly.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Practical Considerations
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Practical Considerations for Polynomial Models</h3>", unsafe_allow_html=True)

	st.markdown("""
    ### When to Use Polynomial Terms

    Consider using polynomial terms when:

    1. Theory suggests a non-linear relationship
    2. Exploratory data analysis reveals a non-linear pattern
    3. A scatter plot of residuals vs. the variable shows a non-random pattern
    4. You need to model a relationship with a maximum or minimum

    ### Calculating and Reporting Marginal Effects

    When working with polynomial models, it's important to calculate the marginal effect at meaningful values of $X$:

    1. At the mean of $X$
    2. At several points across the range of $X$ (e.g., at the 25th, 50th, and 75th percentiles)
    3. At theoretically important values

    These calculated marginal effects should be reported along with their standard errors and confidence intervals.
    """)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Practical Tips for Polynomial Models:**

    1. Center variables before creating polynomial terms to reduce multicollinearity and make the coefficients more interpretable
    2. Consider using orthogonal polynomials for better numerical stability
    3. Plot the fitted curve and marginal effects to aid interpretation
    4. Don't use unnecessarily high-order polynomials - usually quadratic or cubic is sufficient
    5. Be cautious about extrapolating polynomial models far beyond the range of your data
    6. Consider alternative non-linear specifications like splines or logarithmic transformations
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

# Practical Examples
elif section == "Practical Examples":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Practical Examples of Coefficient Interpretation</h2>", unsafe_allow_html=True)

	st.markdown("""
    This section provides realistic examples of how to interpret coefficients in various contexts, using datasets from economics, social sciences, health, and business applications.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Example 1: Income and Education
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Example 1: Income and Education (Log-Level Model)</h3>", unsafe_allow_html=True)

	st.markdown("""
    Suppose we have the following regression model estimating the relationship between education and income:

    $\\ln(Income) = 9.5 + 0.11 \\times Education + 0.35 \\times Experience - 0.005 \\times Experience^2$

    where $Income$ is annual income in dollars, $Education$ is years of schooling, and $Experience$ is years of work experience.
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Interpretation:**

    **Education coefficient (0.11)**: Each additional year of education is associated with approximately a $(e^{0.11} - 1) \\times 100\\% = 11.6\\%$ increase in income, holding experience constant.

    **Experience coefficient (0.35)**: The effect of an additional year of experience depends on how much experience one already has:

    $\\frac{\\partial \\ln(Income)}{\\partial Experience} = 0.35 - 2 \\times 0.005 \\times Experience = 0.35 - 0.01 \\times Experience$

    - At Experience = 0, an additional year increases income by approximately 35%.
    - At Experience = 10, an additional year increases income by approximately 0.35 - 0.01 × 10 = 0.25 or 25%.
    - At Experience = 35, the marginal effect is zero (peak earnings).
    - Beyond 35 years, additional experience is associated with a decrease in income.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Create sample data for visualization
	np.random.seed(42)
	education = np.random.randint(8, 22, 200)
	experience = np.random.randint(0, 40, 200)
	log_income = 9.5 + 0.11 * education + 0.35 * experience - 0.005 * experience ** 2 + np.random.normal(0, 0.3, 200)
	income = np.exp(log_income)

	df = pd.DataFrame({
		'Education': education,
		'Experience': experience,
		'Income': income
	})

	# Create plot for education effect
	fig1 = px.scatter(df, x='Education', y='Income', opacity=0.7,
					  title='Income vs. Education (with fixed Experience = 15 years)')

	edu_range = np.linspace(8, 22, 100)
	income_pred = np.exp(9.5 + 0.11 * edu_range + 0.35 * 15 - 0.005 * 15 ** 2)

	fig1.add_trace(
		go.Scatter(x=edu_range, y=income_pred, mode='lines',
				   name='Predicted Income',
				   line=dict(color='red', width=3))
	)

	# Create plot for experience effect
	fig2 = px.scatter(df, x='Experience', y='Income', opacity=0.7,
					  title='Income vs. Experience (with fixed Education = 16 years)')

	exp_range = np.linspace(0, 40, 100)
	income_pred2 = np.exp(9.5 + 0.11 * 16 + 0.35 * exp_range - 0.005 * exp_range ** 2)

	fig2.add_trace(
		go.Scatter(x=exp_range, y=income_pred2, mode='lines',
				   name='Predicted Income',
				   line=dict(color='red', width=3))
	)

	# Add annotation for peak earnings
	peak_exp = 35
	peak_income = np.exp(9.5 + 0.11 * 16 + 0.35 * peak_exp - 0.005 * peak_exp ** 2)

	fig2.add_trace(
		go.Scatter(x=[peak_exp], y=[peak_income],
				   mode='markers',
				   marker=dict(size=12, color='green', symbol='star'),
				   name='Peak earnings at 35 years')
	)

	col1, col2 = st.columns(2)

	with col1:
		st.plotly_chart(fig1, use_container_width=True)

	with col2:
		st.plotly_chart(fig2, use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Example 2: House Prices
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Example 2: House Prices (Log-Log Model with Dummy)</h3>",
				unsafe_allow_html=True)

	st.markdown("""
    Consider a model of house prices with the following specification:

    $\\ln(Price) = 12.1 + 1.2 \\times \\ln(Size) - 0.05 \\times Age + 0.3 \\times Garage + 0.2 \\times (Garage \\times \\ln(Size))$

    where $Price$ is in dollars, $Size$ is square footage, $Age$ is in years, and $Garage$ is a dummy variable (1 if the house has a garage, 0 otherwise).
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Interpretation:**

    **Size coefficient (1.2)**: For houses without a garage (Garage = 0), a 1% increase in size is associated with a 1.2% increase in price. This is the price elasticity with respect to size for houses without garages.

    **Age coefficient (-0.05)**: A one-year increase in age is associated with approximately a $(e^{-0.05} - 1) \\times 100\\% = -4.9\\%$ change in price, holding other variables constant. In other words, each additional year of age reduces the house price by about 4.9%.

    **Garage coefficient (0.3)**: For houses of size = 1 sq.ft. (theoretical), having a garage is associated with a $(e^{0.3} - 1) \\times 100\\% = 35\\%$ higher price. However, because of the interaction term, the effect of having a garage depends on the size of the house.

    **Interaction coefficient (0.2)**: The difference in the price elasticity with respect to size between houses with and without garages. For houses with a garage (Garage = 1), a 1% increase in size is associated with a (1.2 + 0.2) = 1.4% increase in price.

    The total effect of having a garage for a house of size $S$ is:

    $\\Delta \\ln(Price) = 0.3 + 0.2 \\times \\ln(S)$

    For example, for a 2,000 sq.ft. house, having a garage is associated with a price that is approximately $(e^{0.3 + 0.2 \\times \\ln(2000)} - 1) \\times 100\\% = (e^{0.3 + 0.2 \\times 7.6} - 1) \\times 100\\% = (e^{1.82} - 1) \\times 100\\% = 517\\%$ higher. This extreme result suggests the model may be misspecified or extrapolating beyond reasonable ranges.
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Create sample data for visualization
	np.random.seed(42)
	size = np.random.uniform(1000, 3000, 200)
	age = np.random.randint(0, 50, 200)
	garage = np.random.binomial(1, 0.7, 200)
	log_price = 12.1 + 1.2 * np.log(size) - 0.05 * age + 0.3 * garage + 0.2 * (
				garage * np.log(size)) + np.random.normal(0, 0.2, 200)
	price = np.exp(log_price)

	df2 = pd.DataFrame({
		'Size': size,
		'Age': age,
		'Garage': garage,
		'Price': price,
		'Garage_Category': ['With Garage' if g == 1 else 'No Garage' for g in garage]
	})

	# Create plots
	fig3 = px.scatter(df2, x='Size', y='Price', color='Garage_Category', opacity=0.7,
					  title='House Price vs. Size by Garage Status (Age = 20 years)',
					  color_discrete_sequence=['blue', 'red'])

	size_range = np.linspace(1000, 3000, 100)

	# Predicted price for houses without garage
	price_pred_no_garage = np.exp(
		12.1 + 1.2 * np.log(size_range) - 0.05 * 20 + 0.3 * 0 + 0.2 * (0 * np.log(size_range)))

	# Predicted price for houses with garage
	price_pred_garage = np.exp(12.1 + 1.2 * np.log(size_range) - 0.05 * 20 + 0.3 * 1 + 0.2 * (1 * np.log(size_range)))

	fig3.add_trace(
		go.Scatter(x=size_range, y=price_pred_no_garage, mode='lines',
				   name='Predicted (No Garage)',
				   line=dict(color='blue', width=3))
	)

	fig3.add_trace(
		go.Scatter(x=size_range, y=price_pred_garage, mode='lines',
				   name='Predicted (With Garage)',
				   line=dict(color='red', width=3))
	)

	# Plot for age effect
	fig4 = px.scatter(df2, x='Age', y='Price', color='Garage_Category', opacity=0.7,
					  title='House Price vs. Age by Garage Status (Size = 2000 sq.ft.)',
					  color_discrete_sequence=['blue', 'red'])

	age_range = np.linspace(0, 50, 100)

	# Predicted price for houses without garage
	price_pred_no_garage2 = np.exp(12.1 + 1.2 * np.log(2000) - 0.05 * age_range + 0.3 * 0 + 0.2 * (0 * np.log(2000)))

	# Predicted price for houses with garage
	price_pred_garage2 = np.exp(12.1 + 1.2 * np.log(2000) - 0.05 * age_range + 0.3 * 1 + 0.2 * (1 * np.log(2000)))

	fig4.add_trace(
		go.Scatter(x=age_range, y=price_pred_no_garage2, mode='lines',
				   name='Predicted (No Garage)',
				   line=dict(color='blue', width=3))
	)

	fig4.add_trace(
		go.Scatter(x=age_range, y=price_pred_garage2, mode='lines',
				   name='Predicted (With Garage)',
				   line=dict(color='red', width=3))
	)

	col1, col2 = st.columns(2)

	with col1:
		st.plotly_chart(fig3, use_container_width=True)

	with col2:
		st.plotly_chart(fig4, use_container_width=True)

	st.markdown("<div class='note'>", unsafe_allow_html=True)
	st.markdown("""
    **Note:** This example illustrates that models with interactions and transformations can lead to extreme predictions when extrapolating. It's important to validate model predictions and consider whether they make practical sense.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

	# Example 3: Environmental model
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h3 class='sub-header'>Example 3: Environmental Model (Polynomial Terms)</h3>", unsafe_allow_html=True)

	st.markdown("""
    Consider an environmental study on the relationship between plant growth and temperature:

    $Growth = -20 + 3 \\times Temp - 0.05 \\times Temp^2$

    where $Growth$ is measured in millimeters per week and $Temp$ is temperature in degrees Celsius.
    """)

	st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
	st.markdown("""
    **Interpretation:**

    This is a quadratic model where the effect of temperature on plant growth depends on the temperature itself.

    The marginal effect of temperature on growth is:

    $\\frac{\\partial Growth}{\\partial Temp} = 3 - 0.1 \\times Temp$

    - At Temp = 0°C, an increase of 1°C is associated with a 3 mm/week increase in growth.
    - At Temp = 20°C, an increase of 1°C is associated with a 3 - 0.1 × 20 = 1 mm/week increase in growth.
    - At Temp = 30°C, an increase of 1°C is associated with a 3 - 0.1 × 30 = 0 mm/week change in growth (no effect).
    - Beyond 30°C, increases in temperature are associated with decreased growth.

    The temperature that maximizes growth is:

    $Temp_{max} = -\\frac{3}{2 \\times (-0.05)} = 30°C$

    At this temperature, the predicted growth rate is:

    $Growth_{max} = -20 + 3 \\times 30 - 0.05 \\times 30^2 = -20 + 90 - 45 = 25$ mm/week
    """)
	st.markdown("</div>", unsafe_allow_html=True)

	# Create sample data for visualization
	np.random.seed(42)
	temp = np.random.uniform(5, 45, 200)
	growth = -20 + 3 * temp - 0.05 * temp ** 2 + np.random.normal(0, 3, 200)

	df3 = pd.DataFrame({
		'Temperature': temp,
		'Growth': growth
	})

	# Create plot
	fig5 = px.scatter(df3, x='Temperature', y='Growth', opacity=0.7,
					  title='Plant Growth vs. Temperature')

	temp_range = np.linspace(0, 50, 100)
	growth_pred = -20 + 3 * temp_range - 0.05 * temp_range ** 2

	fig5.add_trace(
		go.Scatter(x=temp_range, y=growth_pred, mode='lines',
				   name='Predicted Growth',
				   line=dict(color='green', width=3))
	)

	# Add annotation for optimal temperature
	optimal_temp = 30
	max_growth = -20 + 3 * optimal_temp - 0.05 * optimal_temp ** 2

	fig5.add_trace(
		go.Scatter(x=[optimal_temp], y=[max_growth],
				   mode='markers',
				   marker=dict(size=12, color='red', symbol='star'),
				   name='Optimal temperature (30°C)')
	)

	# Add marginal effects at different temperatures
	temps_to_mark = [10, 20, 30, 40]

	for t in temps_to_mark:
		growth_at_t = -20 + 3 * t - 0.05 * t ** 2
		slope_at_t = 3 - 0.1 * t

		fig5.add_trace(
			go.Scatter(x=[t - 2, t + 2],
					   y=[growth_at_t - slope_at_t * 2, growth_at_t + slope_at_t * 2],
					   mode='lines',
					   line=dict(color='blue', width=2, dash='dash'),
					   name=f'Slope at {t}°C: {slope_at_t:.1f}')
		)

	# Create a second plot for marginal effects
	fig6 = go.Figure()

	fig6.add_trace(
		go.Scatter(x=temp_range, y=3 - 0.1 * temp_range, mode='lines',
				   name='Marginal Effect of Temperature',
				   line=dict(color='purple', width=3))
	)

	fig6.add_trace(
		go.Scatter(x=[0, 50], y=[0, 0], mode='lines',
				   name='Zero Effect Line',
				   line=dict(color='red', width=2, dash='dash'))
	)

	fig6.update_layout(
		title='Marginal Effect of Temperature on Growth',
		xaxis_title='Temperature (°C)',
		yaxis_title='Marginal Effect (mm/week per 1°C)'
	)

	col1, col2 = st.columns(2)

	with col1:
		st.plotly_chart(fig5, use_container_width=True)

	with col2:
		st.plotly_chart(fig6, use_container_width=True)

	st.markdown("<div class='tip'>", unsafe_allow_html=True)
	st.markdown("""
    **Tip:** For polynomial models, always plot:
    1. The fitted curve to visualize the relationship
    2. The marginal effects to see how the impact changes across the range of the predictor

    These plots make the interpretation much clearer than just looking at the coefficients.
    """)
	st.markdown("</div>", unsafe_allow_html=True)
	st.markdown("</div>", unsafe_allow_html=True)

# Interactive Playground
elif section == "Interactive Playground":
	st.markdown("<div class='section'>", unsafe_allow_html=True)
	st.markdown("<h2 class='sub-header'>Interactive Regression Coefficient Playground</h2>", unsafe_allow_html=True)

	st.markdown("""
    Explore how changes in regression coefficients affect model predictions and interpretations. Choose a model type and adjust the parameters to see the effects in real-time.
    """)

	model_type = st.selectbox(
		"Select Model Type:",
		["Linear (Level-Level)", "Log-Level", "Level-Log", "Log-Log",
		 "Model with Dummy Variable", "Quadratic Model", "Cubic Model"]
	)

	st.markdown("### Set Model Parameters")

	if model_type == "Linear (Level-Level)":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
			slope = st.slider("Slope (β₁)", -5.0, 5.0, 3.0, 0.1)

		with col2:
			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $Y = {intercept:.1f} + {slope:.1f} X$

            **Interpretation:**
            - Intercept ({intercept:.1f}): The expected value of Y when X = 0
            - Slope ({slope:.1f}): A one-unit increase in X is associated with a {slope:.1f} unit change in Y
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0, 10, 100)
		y = intercept + slope * x

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y, mode='lines',
					   name='Model Prediction',
					   line=dict(color='blue', width=3))
		)

		# Highlight the interpretation of slope
		x_point = 5
		y_point = intercept + slope * x_point

		fig.add_trace(
			go.Scatter(x=[x_point, x_point + 1],
					   y=[y_point, y_point + slope],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name='Δx=1 → Δy=' + str(slope))
		)

		fig.update_layout(
			title="Linear Model",
			xaxis_title="X",
			yaxis_title="Y",
			height=500
		)

		st.plotly_chart(fig, use_container_width=True)

	elif model_type == "Log-Level":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -5.0, 5.0, 1.0, 0.1)
			slope = st.slider("Slope (β₁)", -0.5, 0.5, 0.2, 0.01)

		with col2:
			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $\\ln(Y) = {intercept:.1f} + {slope:.2f} X$

            **Interpretation:**
            - A one-unit increase in X is associated with a {(np.exp(slope) - 1) * 100:.1f}% change in Y
              * Calculation: $(e^{{{slope:.2f}}} - 1) \\times 100\\% = {(np.exp(slope) - 1) * 100:.1f}\\%$

            - For small values of β₁, approximately a {slope * 100:.1f}% change in Y
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0, 10, 100)
		log_y = intercept + slope * x
		y = np.exp(log_y)

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y, mode='lines',
					   name='Model Prediction',
					   line=dict(color='blue', width=3))
		)

		# Highlight the interpretation at a point
		x_point = 5
		y_point = np.exp(intercept + slope * x_point)
		x_point_plus = x_point + 1
		y_point_plus = np.exp(intercept + slope * x_point_plus)

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_plus],
					   y=[y_point, y_point_plus],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name=f'Δx=1 → %Δy={((y_point_plus / y_point) - 1) * 100:.1f}%')
		)

		fig.update_layout(
			title="Log-Level Model",
			xaxis_title="X",
			yaxis_title="Y",
			height=500
		)

		st.plotly_chart(fig, use_container_width=True)

	elif model_type == "Level-Log":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
			slope = st.slider("Slope (β₁)", -10.0, 10.0, 3.0, 0.1)

		with col2:
			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $Y = {intercept:.1f} + {slope:.1f} \\ln(X)$

            **Interpretation:**
            - A 1% increase in X is associated with a {slope / 100:.3f} unit change in Y
              * Calculation: $\\frac{{{slope:.1f}}}{{100}} = {slope / 100:.3f}$ units

            - Doubling X (100% increase) is associated with a {slope * np.log(2):.2f} unit change in Y
              * Calculation: ${slope:.1f} \\times \\ln(2) = {slope * np.log(2):.2f}$ units
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0.1, 10, 100)  # Start from 0.1 to avoid log(0)
		y = intercept + slope * np.log(x)

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y, mode='lines',
					   name='Model Prediction',
					   line=dict(color='blue', width=3))
		)

		# Highlight the interpretation at a point
		x_point = 5
		y_point = intercept + slope * np.log(x_point)
		x_point_plus = x_point * 1.01  # 1% increase
		y_point_plus = intercept + slope * np.log(x_point_plus)

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_plus],
					   y=[y_point, y_point_plus],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name=f'1% increase in X → Δy={y_point_plus - y_point:.4f}')
		)

		# Also show doubling
		x_double = x_point * 2
		y_double = intercept + slope * np.log(x_double)

		fig.add_trace(
			go.Scatter(x=[x_point, x_double],
					   y=[y_point, y_double],
					   mode='lines+markers',
					   marker=dict(size=10, color='red'),
					   line=dict(width=4, color='red', dash='dot'),
					   name=f'Doubling X → Δy={y_double - y_point:.2f}')
		)

		fig.update_layout(
			title="Level-Log Model",
			xaxis_title="X",
			yaxis_title="Y",
			height=500
		)

		st.plotly_chart(fig, use_container_width=True)

	elif model_type == "Log-Log":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -5.0, 5.0, 1.0, 0.1)
			elasticity = st.slider("Elasticity (β₁)", 0.0, 2.0, 0.7, 0.05)

		with col2:
			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $\\ln(Y) = {intercept:.1f} + {elasticity:.2f} \\ln(X)$

            **Interpretation:**
            - β₁ = {elasticity:.2f} is the elasticity of Y with respect to X
            - A 1% increase in X is associated with a {elasticity:.2f}% increase in Y
            - A 10% increase in X is associated with a {elasticity * 10:.1f}% increase in Y
            - Doubling X (100% increase) is associated with a {elasticity * 100:.1f}% increase in Y
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0.1, 10, 100)  # Start from 0.1 to avoid log(0)
		log_y = intercept + elasticity * np.log(x)
		y = np.exp(log_y)

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y, mode='lines',
					   name='Model Prediction',
					   line=dict(color='blue', width=3))
		)

		# Highlight the interpretation at a point
		x_point = 5
		y_point = np.exp(intercept + elasticity * np.log(x_point))

		# 1% increase
		x_point_plus = x_point * 1.01
		y_point_plus = np.exp(intercept + elasticity * np.log(x_point_plus))

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_plus],
					   y=[y_point, y_point_plus],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name=f'1% increase in X → %Δy={((y_point_plus / y_point) - 1) * 100:.2f}%')
		)

		# 10% increase
		x_point_10 = x_point * 1.1
		y_point_10 = np.exp(intercept + elasticity * np.log(x_point_10))

		fig.add_trace(
			go.Scatter(x=[x_point, x_point_10],
					   y=[y_point, y_point_10],
					   mode='lines+markers',
					   marker=dict(size=10, color='red'),
					   line=dict(width=4, color='red', dash='dot'),
					   name=f'10% increase in X → %Δy={((y_point_10 / y_point) - 1) * 100:.1f}%')
		)

		fig.update_layout(
			title="Log-Log Model",
			xaxis_title="X",
			yaxis_title="Y",
			height=500
		)

		st.plotly_chart(fig, use_container_width=True)

	elif model_type == "Model with Dummy Variable":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 2.0, 0.5)
			slope = st.slider("Slope for continuous var (β₁)", -5.0, 5.0, 3.0, 0.1)
			dummy_effect = st.slider("Dummy effect (β₂)", -10.0, 10.0, 4.0, 0.5)
			interaction = st.slider("Interaction (β₃)", -3.0, 3.0, -1.5, 0.1)

		with col2:
			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $Y = {intercept:.1f} + {slope:.1f}X + {dummy_effect:.1f}D + {interaction:.1f}(X \\times D)$

            **Interpretation:**

            For D = 0 (reference group):
            - Y = {intercept:.1f} + {slope:.1f}X
            - Each unit increase in X is associated with a {slope:.1f} unit change in Y

            For D = 1:
            - Y = ({intercept:.1f} + {dummy_effect:.1f}) + ({slope:.1f} + {interaction:.1f})X = {intercept + dummy_effect:.1f} + {slope + interaction:.1f}X
            - Each unit increase in X is associated with a {slope + interaction:.1f} unit change in Y

            The coefficient {dummy_effect:.1f} represents the difference in Y between the two groups when X = 0.
            The coefficient {interaction:.1f} represents the difference in slopes between the two groups.
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0, 10, 100)
		y0 = intercept + slope * x  # for D=0
		y1 = (intercept + dummy_effect) + (slope + interaction) * x  # for D=1

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y0, mode='lines',
					   name='D = 0 (Reference)',
					   line=dict(color='blue', width=3))
		)

		fig.add_trace(
			go.Scatter(x=x, y=y1, mode='lines',
					   name='D = 1',
					   line=dict(color='red', width=3))
		)

		# Highlight the interpretation of slopes
		x_point = 5
		y0_point = intercept + slope * x_point
		y1_point = (intercept + dummy_effect) + (slope + interaction) * x_point

		fig.add_trace(
			go.Scatter(x=[x_point, x_point + 1],
					   y=[y0_point, y0_point + slope],
					   mode='lines+markers',
					   marker=dict(size=10, color='green'),
					   line=dict(width=4, color='green', dash='dash'),
					   name=f'Slope for D=0: {slope:.1f}')
		)

		fig.add_trace(
			go.Scatter(x=[x_point, x_point + 1],
					   y=[y1_point, y1_point + (slope + interaction)],
					   mode='lines+markers',
					   marker=dict(size=10, color='orange'),
					   line=dict(width=4, color='orange', dash='dash'),
					   name=f'Slope for D=1: {slope + interaction:.1f}')
		)

		# Highlight the dummy effect at X=0
		fig.add_annotation(
			x=0,
			y=(intercept + dummy_effect / 2),
			text=f"Dummy effect at X=0: {dummy_effect:.1f}",
			showarrow=True,
			arrowhead=2,
			ax=50,
			ay=0
		)

		fig.update_layout(
			title="Model with Dummy Variable and Interaction",
			xaxis_title="X",
			yaxis_title="Y",
			height=500
		)

		st.plotly_chart(fig, use_container_width=True)

	elif model_type == "Quadratic Model":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 2.0, 0.5)
			linear = st.slider("Linear term (β₁)", -5.0, 5.0, 1.5, 0.1)
			quadratic = st.slider("Quadratic term (β₂)", -0.5, 0.5, -0.1, 0.01)

		with col2:
			# Calculate the turning point
			turning_point = -linear / (2 * quadratic) if quadratic != 0 else "undefined"
			max_or_min = "maximum" if quadratic < 0 else "minimum" if quadratic > 0 else "none"

			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $Y = {intercept:.1f} + {linear:.1f}X + {quadratic:.2f}X^2$

            **Interpretation:**
            - Marginal effect of X on Y: $\\frac{{\\partial Y}}{{\\partial X}} = {linear:.1f} + {2 * quadratic:.2f}X$
            - The effect of X on Y depends on the value of X itself
            - The relationship is {'concave (∩)' if quadratic < 0 else 'convex (∪)' if quadratic > 0 else 'linear'}
            - Turning point ({max_or_min}): X = {turning_point if isinstance(turning_point, str) else turning_point:.1f}
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0, 10, 100)
		y = intercept + linear * x + quadratic * x ** 2

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y, mode='lines',
					   name='Model Prediction',
					   line=dict(color='blue', width=3))
		)

		# Highlight the marginal effects at different points
		for x_point in [2, 5, 8]:
			y_point = intercept + linear * x_point + quadratic * x_point ** 2
			slope = linear + 2 * quadratic * x_point

			fig.add_trace(
				go.Scatter(x=[x_point - 0.5, x_point + 0.5],
						   y=[y_point - slope * 0.5, y_point + slope * 0.5],
						   mode='lines',
						   line=dict(width=3, color='green'),
						   name=f'Slope at X={x_point}: {slope:.2f}')
			)

			fig.add_annotation(
				x=x_point,
				y=y_point,
				text=f"Marginal effect: {slope:.2f}",
				showarrow=True,
				arrowhead=2,
				ax=0,
				ay=-40
			)

		# Add turning point if it exists and is within range
		if not isinstance(turning_point, str) and 0 <= turning_point <= 10:
			max_min_y = intercept + linear * turning_point + quadratic * turning_point ** 2

			fig.add_trace(
				go.Scatter(x=[turning_point], y=[max_min_y],
						   mode='markers',
						   marker=dict(size=12, color='red', symbol='star'),
						   name=f'{max_or_min.capitalize()} at X={turning_point:.1f}')
			)

			fig.add_annotation(
				x=turning_point,
				y=max_min_y,
				text=f"{max_or_min.capitalize()} at X={turning_point:.1f}",
				showarrow=True,
				arrowhead=2,
				ax=0,
				ay=-60
			)

		fig.update_layout(
			title="Quadratic Model",
			xaxis_title="X",
			yaxis_title="Y",
			height=500
		)

		st.plotly_chart(fig, use_container_width=True)

	elif model_type == "Cubic Model":
		col1, col2 = st.columns(2)

		with col1:
			intercept = st.slider("Intercept (β₀)", -10.0, 10.0, 2.0, 0.5)
			linear = st.slider("Linear term (β₁)", -5.0, 5.0, 0.5, 0.1)
			quadratic = st.slider("Quadratic term (β₂)", -1.0, 1.0, 0.2, 0.05)
			cubic = st.slider("Cubic term (β₃)", -0.1, 0.1, -0.03, 0.01)

		with col2:
			st.markdown("<div class='interpretation'>", unsafe_allow_html=True)
			st.markdown(f"""
            **Model Equation:** $Y = {intercept:.1f} + {linear:.1f}X + {quadratic:.2f}X^2 + {cubic:.3f}X^3$

            **Interpretation:**
            - Marginal effect of X on Y: $\\frac{{\\partial Y}}{{\\partial X}} = {linear:.1f} + {2 * quadratic:.2f}X + {3 * cubic:.3f}X^2$
            - The effect of X on Y is a quadratic function of X
            - For large values of X, the relationship is {'eventually decreasing' if cubic < 0 else 'eventually increasing' if cubic > 0 else 'dominated by the quadratic term'}
            """)
			st.markdown("</div>", unsafe_allow_html=True)

		# Generate data
		x = np.linspace(0, 10, 100)
		y = intercept + linear * x + quadratic * x ** 2 + cubic * x ** 3

		# Create plot
		fig = go.Figure()

		fig.add_trace(
			go.Scatter(x=x, y=y, mode='lines',
					   name='Model Prediction',
					   line=dict(color='blue', width=3))
		)

		# Highlight the marginal effects at different points
		for x_point in [2, 5, 8]:
			y_point = intercept + linear * x_point + quadratic * x_point ** 2 + cubic * x_point ** 3
			slope = linear + 2 * quadratic * x_point + 3 * cubic * x_point ** 2

			fig.add_trace(
				go.Scatter(x=[x_point - 0.5, x_point + 0.5],
						   y=[y_point - slope * 0.5, y_point + slope * 0.5],
						   mode='lines',
						   line=dict(width=3, color='green'),
						   name=f'Slope at X={x_point}: {slope:.2f}')
			)

			fig.add_annotation(
				x=x_point,
				y=y_point,
				text=f"Marginal effect: {slope:.2f}",
				showarrow=True,
				arrowhead=2,
				ax=0,
				ay=-40
			)

		# Create a second plot for marginal effects
		marg_effect = linear + 2 * quadratic * x + 3 * cubic * x ** 2

		fig2 = go.Figure()

		fig2.add_trace(
			go.Scatter(x=x, y=marg_effect, mode='lines',
					   name='Marginal Effect',
					   line=dict(color='purple', width=3))
		)

		fig2.add_trace(
			go.Scatter(x=[0, 10], y=[0, 0], mode='lines',
					   name='Zero Effect Line',
					   line=dict(color='red', width=2, dash='dash'))
		)

		fig2.update_layout(
			title='Marginal Effect of X on Y',
			xaxis_title='X',
			yaxis_title='Marginal Effect (∂Y/∂X)'
		)

		st.plotly_chart(fig, use_container_width=True)
		st.plotly_chart(fig2, use_container_width=True)

	st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
---
### Additional Resources

For more information on interpreting regression coefficients:

- [Wooldridge, J. M. (2020). Introductory Econometrics: A Modern Approach. Cengage Learning.](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/9781337558860/)
- [Angrist, J. D., & Pischke, J. S. (2008). Mostly Harmless Econometrics: An Empiricist's Companion. Princeton University Press.](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
- [Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.](https://www.cambridge.org/core/books/data-analysis-using-regression-and-multilevel-hierarchical-models/32A29531C7FD730C3A68951A17C9D983)

© 2025 Regression Coefficient Interpretation Guide
""")
