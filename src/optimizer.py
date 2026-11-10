from optimizer.data_loader import load_inputs
from optimizer.validation import validate_input_data
from optimizer.selection import select_top_stocks
from optimizer.price_matrix import prepare_price_matrix
from optimizer.returns import calculate_returns
from optimizer.covariance import calculate_covariance
from optimizer.portfolio_optimizer import optimize_portfolio
from optimizer.allocation import calculate_allocation_amounts
from optimizer.output import save_allocations

def run_optimizer():
    predictions, prices = load_inputs(
        predictions_path="../data/pred_oos.parquet",
        prices_path="../data/raw/prices/prices.parquet"
    )

    print("Datasets loaded successfully.")

    # Step 2: Validate datasets
    predictions, prices = validate_input_data(
        predictions,
        prices
    )

    # Step 3: Select Top N stocks
    selected_stocks = select_top_stocks(
        predictions=predictions,
        top_n=3
    )

    selected_tickers = selected_stocks["ticker"].tolist()

    prediction_date = selected_stocks["date"].iloc[0]

    # Step 4: Prepare historical price matrix
    price_matrix = prepare_price_matrix(
        prices=prices,
        selected_tickers=selected_tickers,
        prediction_date=prediction_date,
        lookback_sessions=252,
        price_column="close"
    )

    # Step 5: Calculate historical returns
    returns = calculate_returns(price_matrix)

    # Step 6: Calculate cov matrix
    covariance_matrix = calculate_covariance(returns)

	 # Step 7: Portfolio optimization
    weights = optimize_portfolio(
        covariance_matrix=covariance_matrix,
        max_weight=0.60
    )

	 # Step 8: Allocations
    total_capital = 10_000.00

    allocations = calculate_allocation_amounts(
	     weights=weights,
	     total_capital=total_capital
    )

    #Step 9: Save allocations to allocations.parquet
    save_allocations(
        allocations=allocations,
        rebalance_date=prediction_date,
        output_path="../data/allocations/allocations.parquet",
        strategy="minimum_variance"
    )
    saved_allocations = pd.read_parquet(
        "../data/allocations/allocations.parquet"
    )
