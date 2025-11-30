import json
import uuid
import datetime
import os

# Load product catalog
with open("shared-data/day9_catalog.json", "r") as f:
    PRODUCTS = json.load(f)

# Store orders in memory + file
ORDERS_FILE = "orders/day9_orders.json"
os.makedirs("orders", exist_ok=True)

if os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "r") as f:
        ORDERS = json.load(f)
else:
    ORDERS = []


# -----------------------------------
# Helper: Save orders to JSON
# -----------------------------------
def _save_orders():
    with open(ORDERS_FILE, "w") as f:
        json.dump(ORDERS, f, indent=2)


# -----------------------------------
# PRODUCT LISTING (ACP-style filtering)
# -----------------------------------
def list_products(filters=None):
    """
    filters example:
    {
        "category": "hoodies",
        "max_price": 1000,
        "color": "black",
        "keyword": "mug"
    }
    """

    results = PRODUCTS

    if not filters:
        return results

    if "category" in filters:
        results = [p for p in results if p["category"] == filters["category"]]

    if "max_price" in filters:
        results = [p for p in results if p["price"] <= filters["max_price"]]

    if "color" in filters:
        results = [p for p in results if filters["color"] in p.get("color", [])]

    if "size" in filters:
        results = [p for p in results if filters["size"] in p.get("sizes", [])]

    if "keyword" in filters:
        keyword = filters["keyword"].lower()
        results = [p for p in results
                   if keyword in p["name"].lower()
                   or keyword in p["description"].lower()
                   or keyword in p["category"].lower()]

    return results


# -----------------------------------
# CREATE ORDER (ACP-style)
# -----------------------------------
def create_order(line_items):
    """
    line_items example:
    [
        {"product_id": "hoodie-001", "quantity": 1},
        {"product_id": "mug-002", "quantity": 2}
    ]
    """

    order_items = []
    total = 0
    currency = "INR"

    for item in line_items:
        pid = item["product_id"]
        qty = item.get("quantity", 1)

        # Find product
        product = next((p for p in PRODUCTS if p["id"] == pid), None)
        if not product:
            raise ValueError(f"Product ID '{pid}' does not exist.")

        subtotal = product["price"] * qty

        order_items.append({
            "product_id": pid,
            "name": product["name"],
            "unit_amount": product["price"],
            "quantity": qty,
            "subtotal": subtotal
        })

        total += subtotal

    order = {
        "id": str(uuid.uuid4())[:8],
        "items": order_items,
        "total": total,
        "currency": currency,
        "created_at": datetime.datetime.utcnow().isoformat()
    }

    # Save
    ORDERS.append(order)
    _save_orders()

    return order


# -----------------------------------
# GET LAST ORDER
# -----------------------------------
def get_last_order():
    if not ORDERS:
        return None
    return ORDERS[-1]
