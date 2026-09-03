import reflex as rx

from frontend.State.state import NotebookState
from frontend.pages.home import home
from frontend.pages.create import create_page

style = {
    ":root": {
        "--bg-color": "#0A0A10",
        "--surface-color": "rgba(20, 20, 30, 0.7)",
        "--accent-color": "linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%)",
        "--text-primary": "#FFFFFF",
        "--text-secondary": "#9CA3AF",
    },
    "body": {
        "background_color": "var(--bg-color)",
        "color": "var(--text-primary)",
        "font_family": "'Outfit', sans-serif",
        "margin": "0",
    }
}

app = rx.App(
    style=style,
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap",
    ],
)

app.add_page(
    home,
    route="/",
    on_load=NotebookState.load_notebooks,
)

app.add_page(
    create_page,
    route="/create",
)