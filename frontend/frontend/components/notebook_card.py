import reflex as rx
from frontend.State.state import NotebookState

def notebook_card(notebook: dict) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.box(
                    rx.icon(
                        "notebook",
                        size=20,
                        color="#ffffff",
                    ),
                    padding="0.65rem",
                    border_radius="0.7rem",
                    background="linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%)",
                    border="1px solid rgba(255, 255, 255, 0.05)",
                ),
                rx.spacer(),
                rx.dropdown_menu.root(
                    rx.dropdown_menu.trigger(
                        rx.icon_button(
                            rx.icon(
                                "ellipsis",
                                size=18,
                            ),
                            variant="ghost",
                            color="var(--text-secondary)",
                            _hover={"background": "rgba(255,255,255,0.1)", "color": "white"}
                        ),
                    ),
                    rx.dropdown_menu.content(
                        rx.dropdown_menu.item(
                            "Rename",
                        ),
                        rx.dropdown_menu.item(
                            "Delete",
                            on_click=lambda: NotebookState.delete_notebook(
                                notebook["id"]
                            ),
                        ),
                    ),
                ),
                width="100%",
                align="center",
            ),
            rx.vstack(
                rx.text(
                    notebook["title"],
                    size="4",
                    weight="bold",
                    color="var(--text-primary)",
                ),
                rx.cond(
                    notebook["description"],
                    rx.text(
                        notebook["description"],
                        size="2",
                        color="var(--text-secondary)",
                        overflow="hidden",
                    ),
                    rx.text(
                        "No description",
                        size="2",
                        color="rgba(255, 255, 255, 0.2)",
                    ),
                ),
                align="start",
                spacing="2",
            ),
            rx.hstack(
                rx.text(
                    "0 tags",
                    size="1",
                    color="rgba(255, 255, 255, 0.3)",
                ),
                width="100%",
            ),
            spacing="4",
            align="stretch",
        ),
        padding="1.25rem",
        border="1px solid rgba(255, 255, 255, 0.05)",
        border_radius="1rem",
        background="var(--surface-color)",
        backdrop_filter="blur(16px)",
        _hover={
            "border_color": "rgba(139, 92, 246, 0.4)",
            "box_shadow": "0 10px 25px -5px rgba(139, 92, 246, 0.25)",
            "transform": "translateY(-4px)",
        },
        transition="all 0.3s ease",
    )
