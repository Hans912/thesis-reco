"""Chatbot logic — GPT-4o-mini with tool calling for product recommendations."""

from __future__ import annotations

import base64
import json
import math
import tempfile
from typing import Any, Optional

from openai import OpenAI

from pipelines.search import search_by_image, search_by_text

MIN_SCORE_THRESHOLD = 0.10

SYSTEM_PROMPT = (
    "You are a multilingual product recommendation assistant for Arcaplanet "
    "(pet supplies) and Twinset (fashion). Help users find products by asking "
    "clarifying questions, then search the catalog. Be concise and helpful. "
    "You can respond in whatever language the user writes in.\n\n"
    "After searching, you MUST call select_products with the product_ids of "
    "only the products you genuinely recommend. Do not include weak or "
    "irrelevant matches — it is better to recommend 3 great products than "
    "10 mediocre ones. The products you select will be displayed to the user "
    "as a carousel, so only select products you would confidently recommend.\n\n"
    "After recommending products, you SHOULD also call find_nearby_stores for "
    "the relevant merchant(s) so the user can see where to buy them. A map with "
    "store locations will be displayed alongside the product carousel."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the product catalog using a natural-language query. "
                "Returns candidate products ranked by similarity. You must then "
                "call select_products to choose which ones to recommend."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive search query (e.g. 'dry cat food for indoor cats')",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price filter in EUR",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price filter in EUR",
                    },
                    "merchant": {
                        "type": "string",
                        "enum": ["arcaplanet", "twinset"],
                        "description": "Filter by merchant",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_image",
            "description": (
                "Search for visually similar products using an uploaded image. "
                "You must then call select_products to choose which ones to recommend."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_products",
            "description": (
                "Select which products from the search results to recommend to the user. "
                "Only include products that are genuinely relevant and good matches. "
                "The selected products will be shown as a carousel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product_id values to recommend",
                    },
                },
                "required": ["product_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_preferences",
            "description": (
                "Present follow-up questions to the user to refine the search. "
                "Use this when you need more information about budget, brand, "
                "category, or specific needs before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of clarifying questions to ask the user",
                    },
                },
                "required": ["questions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_recommendation",
            "description": (
                "Explain why a specific product was recommended. Provide a "
                "plain-language explanation based on the query, similarity, and price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to explain",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearby_stores",
            "description": (
                "Find physical store locations for a merchant. "
                "Can optionally sort by distance from given coordinates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant": {
                        "type": "string",
                        "enum": ["arcaplanet", "twinset"],
                        "description": "Filter stores by merchant",
                    },
                    "lat": {
                        "type": "number",
                        "description": "User latitude for distance sorting",
                    },
                    "lng": {
                        "type": "number",
                        "description": "User longitude for distance sorting",
                    },
                },
                "required": [],
            },
        },
    },
]


def _parse_price(price_str: Optional[str]) -> Optional[float]:
    """Try to parse a price string like '9,29' or '12.50' into a float."""
    if not price_str:
        return None
    try:
        return float(price_str.replace(",", ".").strip().split()[0])
    except (ValueError, IndexError):
        return None


def _filter_by_price(results: list[dict], min_price: float | None, max_price: float | None) -> list[dict]:
    """Filter product dicts by price range."""
    if min_price is None and max_price is None:
        return results
    filtered = []
    for p in results:
        price = _parse_price(p.get("price"))
        if price is None:
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        filtered.append(p)
    return filtered


def _filter_by_score(products: list[dict], threshold: float) -> list[dict]:
    """Remove products below the similarity score threshold."""
    return [p for p in products if p.get("score", 0) >= threshold]


def _df_to_product_list(df, conn) -> list[dict]:
    """Convert search DataFrame to list of product dicts with image URLs."""
    from api.main import _first_image_url

    products = []
    for _, row in df.iterrows():
        img = _first_image_url(conn, row["product_id"])
        products.append({
            "product_id": row["product_id"],
            "merchant": row["merchant"],
            "name": row["name"],
            "price": row.get("price"),
            "currency": row.get("currency"),
            "url": row["url"],
            "image_url": img,
            "score": float(row["score"]),
        })
    return products


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in km between two lat/lng points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _execute_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    app_state: Any,
    image_bytes: bytes | None,
    last_search_context: dict,
) -> tuple[str, list[dict] | None, list[str] | None, list[dict] | None]:
    """Execute a tool call and return (result_text, products, follow_up_questions, stores)."""

    if tool_name == "search_products":
        query = tool_args["query"]
        min_price = tool_args.get("min_price")
        max_price = tool_args.get("max_price")
        merchant = tool_args.get("merchant")

        results_df = search_by_text(
            query, app_state.model, app_state.tokenizer,
            app_state.matrix, app_state.meta, app_state.device,
            top_k=50,
        )

        if merchant:
            results_df = results_df[results_df["merchant"] == merchant]

        products = _df_to_product_list(results_df, app_state.conn)
        products = _filter_by_price(products, min_price, max_price)
        products = _filter_by_score(products, MIN_SCORE_THRESHOLD)
        products = products[:10]

        last_search_context["query"] = query
        last_search_context["candidates"] = {p["product_id"]: p for p in products}

        result_text = json.dumps([
            {"name": p["name"], "price": p["price"], "currency": p["currency"],
             "merchant": p["merchant"], "score": round(p["score"], 4),
             "product_id": p["product_id"]}
            for p in products
        ])
        return result_text, None, None, None

    elif tool_name == "search_by_image":
        if not image_bytes:
            return "No image was provided by the user.", None, None, None

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            results_df = search_by_image(
                tmp.name, app_state.model, app_state.preprocess,
                app_state.matrix, app_state.meta, app_state.device,
                top_k=20,
            )

        products = _df_to_product_list(results_df, app_state.conn)
        products = _filter_by_score(products, MIN_SCORE_THRESHOLD)
        products = products[:10]

        last_search_context["query"] = "(image search)"
        last_search_context["candidates"] = {p["product_id"]: p for p in products}

        result_text = json.dumps([
            {"name": p["name"], "price": p["price"], "currency": p["currency"],
             "merchant": p["merchant"], "score": round(p["score"], 4),
             "product_id": p["product_id"]}
            for p in products
        ])
        return result_text, None, None, None

    elif tool_name == "select_products":
        selected_ids = set(tool_args.get("product_ids", []))
        candidates = last_search_context.get("candidates", {})
        selected = [candidates[pid] for pid in selected_ids if pid in candidates]

        last_search_context["products"] = {p["product_id"]: p for p in selected}

        return json.dumps({"selected": len(selected)}), selected, None, None

    elif tool_name == "ask_preferences":
        questions = tool_args.get("questions", [])
        return json.dumps({"questions": questions}), None, questions, None

    elif tool_name == "explain_recommendation":
        product_id = tool_args.get("product_id", "")
        product = (
            last_search_context.get("products", {}).get(product_id)
            or last_search_context.get("candidates", {}).get(product_id)
        )
        query = last_search_context.get("query", "unknown")

        if product:
            info = (
                f"Product: {product['name']} | Price: {product.get('price', 'N/A')} "
                f"{product.get('currency', '')} | Merchant: {product['merchant']} | "
                f"Similarity score: {product.get('score', 'N/A')} | "
                f"Search query used: \"{query}\""
            )
        else:
            info = f"Product {product_id} not found in recent search results."

        return info, None, None, None

    elif tool_name == "find_nearby_stores":
        merchant = tool_args.get("merchant")
        user_lat = tool_args.get("lat")
        user_lng = tool_args.get("lng")

        conn = app_state.conn
        if merchant:
            rows = conn.execute(
                "SELECT store_id, merchant, display_name, lat, lng, street, street_number, zip_code "
                "FROM stores WHERE merchant = ?",
                (merchant,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT store_id, merchant, display_name, lat, lng, street, street_number, zip_code "
                "FROM stores",
            ).fetchall()

        stores = []
        for r in rows:
            store = {
                "store_id": r[0], "merchant": r[1], "display_name": r[2],
                "lat": r[3], "lng": r[4],
                "address": f"{r[5] or ''} {r[6] or ''}, {r[7] or ''}".strip(", "),
            }
            if user_lat is not None and user_lng is not None:
                store["distance_km"] = round(_haversine_km(user_lat, user_lng, r[3], r[4]), 1)
            stores.append(store)

        if user_lat is not None and user_lng is not None:
            stores.sort(key=lambda s: s["distance_km"])
            stores = stores[:10]
        else:
            stores = stores[:20]

        result_text = json.dumps(stores)
        return result_text, None, None, stores

    return f"Unknown tool: {tool_name}", None, None, None


async def chat(
    messages: list[dict],
    image_bytes: bytes | None,
    app_state: Any,
) -> dict:
    """Run a chat turn with OpenAI tool calling.

    Returns dict with keys: message, products, follow_up_questions.
    """
    client = OpenAI()

    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        openai_messages.append({"role": role, "content": content})

    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        openai_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "I uploaded this image. Find similar products."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        })

    products = None
    follow_up_questions = None
    stores = None
    last_search_context: dict = {}

    max_iterations = 8
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
            openai_messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                result_text, tool_products, tool_questions, tool_stores = _execute_tool(
                    fn_name, fn_args, app_state, image_bytes, last_search_context,
                )

                if tool_products is not None:
                    products = tool_products
                if tool_questions is not None:
                    follow_up_questions = tool_questions
                if tool_stores is not None:
                    stores = tool_stores

                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
        else:
            return {
                "message": choice.message.content or "",
                "products": products,
                "follow_up_questions": follow_up_questions,
                "stores": stores,
            }

    last_content = response.choices[0].message.content or "I encountered an issue. Please try again."
    return {
        "message": last_content,
        "products": products,
        "follow_up_questions": follow_up_questions,
        "stores": stores,
    }
