# Categorize columns by type
data_cols_types = {
    "numeric": [
        "Days_from_order_to_shipment",
        "Days_from_shipment_to_delivery",
        "Days_from_order_to_delivery",
        "Days_between_estimated_and_actual_arrival",
        "Delivery_distance"
    ],
    "textual": [
        "Order_ID",
        "Product_Category",
        "Mode_Of_Transport",
        "Priority",
        "Carrier_name",
        "Warehouse",
        "Supplier_Name",
        "Customer_Name",
        "Status"
    ],
    "date": [
        "Date",
        "Order_date",
        "Start_Shipping_Date",
        "Estimated_Arrival_Date",
        "Actual_Arrival_Date"
    ]
}