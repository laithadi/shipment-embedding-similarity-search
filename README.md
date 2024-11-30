## Introduction & Data Analysis

The dataset contains information about a list of orders. Each row provides details related to different aspects of an order's lifecycle.

### Columns:
- **Date columns**:
  - `Date`
  - `Order_date`
  - `Start_Shipping_Date`
  - `Estimated_Arrival_Date`
  - `Actual_Arrival_Date`

- **Categorical columns**:
  - `Order_ID`
  - `Mode_Of_Transport`
  - `Product_Category`
  - `Priority`
  - `Carrier_name`
  - `Warehouse`
  - `Supplier_Name`
  - `Customer_Name`
  - `Status`

- **Numerical columns**:
  - `Delivery_distance`
  - `Days_from_order_to_shipment`
  - `Days_from_shipment_to_delivery`
  - `Days_from_order_to_delivery`
  - `Days_between_estimated_and_actual_arrival`

### Key Data Insights:
- All orders in the dataset are placed on the same date: **2023-07-03** (as indicated in the `Order_date` column).
- Since all orders are placed on the same date, the `Start_Shipping_Date` is also uniform: **2023-07-03**. This suggests that each order took the same amount of time to process.
- **Unique Order IDs**: There are only 14 unique order IDs (`Order_ID`), meaning that each order may appear multiple times in the dataset.
- Each row for the same `Order_ID` corresponds to a different **date** (`Date`), reflecting the status of the delivery at that point in time.
- The dataset also provides additional details such as:
  - **Product Category** (e.g., Apparel, Groceries)
  - **Mode of Transport** (e.g., Air Freight)
  - **Priority** (e.g., Low, Medium, High)
  - **Carrier**, **Supplier**, **Customer Name**
  - **Warehouse Location**
  - **Delivery Distance**
  - **Order Status** (e.g., Delivered, Pending)

For more detailed exploratory data analysis (EDA), check out [eda.ipynb](/exploratory_data_analysis/eda.ipynb).

[
    "ipykernel>=6.29.5",
    "matplotlib>=3.9.3",
    "pandas>=2.2.3",
    "pytest>=8.3.3",
    "seaborn>=0.13.2",
    "torch>=2.5.1",
    "transformers>=4.46.3",
]