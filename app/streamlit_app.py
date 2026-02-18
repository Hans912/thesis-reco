"""Streamlit frontend for the multimodal product recommendation system."""

from __future__ import annotations

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Thesis Reco", layout="wide")


# ── Helpers ──────────────────────────────────────────────────────────────

def api_get(endpoint: str, params: dict | None = None):
    """GET request to the FastAPI backend."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Is the server running on port 8000?")
        st.stop()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        st.stop()


def api_post_image(endpoint: str, file_bytes: bytes, filename: str, params: dict | None = None):
    """POST an image file to the FastAPI backend."""
    try:
        r = requests.post(
            f"{API_BASE}{endpoint}",
            files={"file": (filename, file_bytes, "image/jpeg")},
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot connect to API. Is the server running on port 8000?")
        st.stop()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        st.stop()


def render_product_grid(products: list[dict], cols: int = 4, show_score: bool = False):
    """Render a grid of product cards."""
    for i in range(0, len(products), cols):
        row = st.columns(cols)
        for j, col in enumerate(row):
            if i + j >= len(products):
                break
            p = products[i + j]
            with col:
                if p.get("image_url"):
                    st.image(API_BASE + p["image_url"], use_container_width=True)
                else:
                    st.markdown("*No image*")

                st.markdown(f"**{p['name'][:80]}**")

                price_str = f"{p['price']} {p.get('currency', '')}" if p.get("price") else "N/A"
                merchant_label = p["merchant"].capitalize()
                st.caption(f"{price_str} | {merchant_label}")

                if show_score and "score" in p:
                    st.caption(f"Similarity: {p['score']:.3f}")

                if st.button("Find Similar", key=f"sim_{p['product_id']}_{i+j}"):
                    st.session_state.mode = "similar"
                    st.session_state.selected_product = p["product_id"]
                    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.title("Thesis Reco")
st.sidebar.markdown("Multimodal product recommendations")

mode = st.sidebar.radio(
    "Mode",
    ["Browse", "Text Search", "Image Search"],
    index=["Browse", "Text Search", "Image Search"].index(
        st.session_state.get("mode", "Browse")
    ) if st.session_state.get("mode") in ["Browse", "Text Search", "Image Search"] else 0,
)

# If we're in "similar" mode (triggered by button), show a back button
if st.session_state.get("mode") == "similar":
    if st.sidebar.button("Back"):
        st.session_state.mode = "Browse"
        st.rerun()
else:
    st.session_state.mode = mode

merchant_filter = st.sidebar.radio("Merchant", ["All", "arcaplanet", "twinset"])
merchant_param = None if merchant_filter == "All" else merchant_filter

top_k = st.sidebar.slider("Results", min_value=5, max_value=30, value=10)


# ── Main Area ────────────────────────────────────────────────────────────

current_mode = st.session_state.get("mode", "Browse")

# ── Browse mode
if current_mode == "Browse":
    st.header("Product Catalog")

    page = st.session_state.get("browse_page", 0)
    page_size = 20

    data = api_get("/api/products", {
        "offset": page * page_size,
        "limit": page_size,
        **({"merchant": merchant_param} if merchant_param else {}),
    })

    st.caption(f"Showing {data['offset'] + 1}–{data['offset'] + len(data['products'])} of {data['total']} products")

    render_product_grid(data["products"])

    # Pagination
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if page > 0 and st.button("Previous"):
            st.session_state.browse_page = page - 1
            st.rerun()
    with col2:
        if (page + 1) * page_size < data["total"] and st.button("Next"):
            st.session_state.browse_page = page + 1
            st.rerun()

# ── Text Search mode
elif current_mode == "Text Search":
    st.header("Text Search")

    query = st.text_input("Search products", placeholder="e.g. cibo per gatti, vestido donna")

    if query:
        data = api_get("/api/search/text", {
            "q": query,
            "top_k": top_k,
            **({"merchant": merchant_param} if merchant_param else {}),
        })
        st.caption(f"{len(data['results'])} results")
        render_product_grid(data["results"], show_score=True)

# ── Image Search mode
elif current_mode == "Image Search":
    st.header("Image Search")

    uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(uploaded, caption="Query image", use_container_width=True)

        with col2:
            with st.spinner("Searching…"):
                data = api_post_image("/api/search/image", uploaded.getvalue(), uploaded.name, {
                    "top_k": top_k,
                    **({"merchant": merchant_param} if merchant_param else {}),
                })

            st.caption(f"{len(data['results'])} results")
            render_product_grid(data["results"], show_score=True)

# ── Similar Products mode
elif current_mode == "similar":
    product_id = st.session_state.get("selected_product")
    if not product_id:
        st.warning("No product selected.")
        st.stop()

    # Show the query product
    product = api_get(f"/api/products/{product_id}")
    st.header("Similar Products")

    col1, col2 = st.columns([1, 3])
    with col1:
        if product.get("image_url"):
            st.image(API_BASE + product["image_url"], use_container_width=True)
        st.markdown(f"**{product['name']}**")
        price_str = f"{product['price']} {product.get('currency', '')}" if product.get("price") else "N/A"
        st.caption(f"{price_str} | {product['merchant'].capitalize()}")

    with col2:
        data = api_get(f"/api/products/{product_id}/similar", {
            "top_k": top_k,
            **({"merchant": merchant_param} if merchant_param else {}),
        })
        st.caption(f"{len(data['results'])} similar products")

    render_product_grid(data["results"], show_score=True)
