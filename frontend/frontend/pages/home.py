import reflex as rx
from frontend.State.state import NotebookState
from frontend.components.navbar import navbar
from frontend.components.sidebar import sidebar
from frontend.components.notebook_card import notebook_card

def notebook_grid() -> rx.Component:
    return rx.cond(
        NotebookState.loading,
        rx.center(
            rx.spinner(size="3", color="var(--text-primary)"),
            width="100%",
            padding="5rem",
        ),
        rx.cond(
            NotebookState.notebooks,
            rx.grid(
                rx.foreach(
                    NotebookState.notebooks,
                    notebook_card,
                ),
                columns="3",
                spacing="4",
                width="100%",
            ),
            rx.center(
                rx.vstack(
                    rx.box(
                        rx.icon(
                            "notebook",
                            size=40,
                            stroke_width=1.5,
                            color="white"
                        ),
                        padding="1.2rem",
                        border_radius="1rem",
                        background="linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%)",
                        border="1px solid rgba(255,255,255,0.05)",
                        box_shadow="0 0 20px rgba(139, 92, 246, 0.2)"
                    ),
                    rx.heading(
                        "No notebooks yet",
                        size="6",
                        color="var(--text-primary)",
                    ),
                    rx.text(
                        "Create your first notebook to get started.",
                        color="var(--text-secondary)",
                    ),
                    rx.link(
                        rx.button(
                            rx.icon("plus", size=17),
                            "Create notebook",
                            radius="full",
                            background="var(--accent-color)",
                            color="#ffffff",
                            box_shadow="0 4px 14px 0 rgba(139, 92, 246, 0.39)",
                            transition="all 0.2s ease",
                            _hover={
                                "transform": "translateY(-1px)",
                                "box_shadow": "0 6px 20px rgba(139, 92, 246, 0.5)",
                            },
                        ),
                        href="/create",
                        underline="none",
                    ),
                    align="center",
                    spacing="4",
                ),
                width="100%",
                padding="5rem",
            ),
        ),
    )

def home() -> rx.Component:
    return rx.box(
        navbar(),
        rx.hstack(
            sidebar(),
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.heading(
                                "Your notebooks",
                                size="7",
                                background_image="var(--accent-color)",
                                background_clip="text",
                                color="transparent",
                                style={"-webkit-background-clip": "text"},
                            ),
                            rx.text(
                                "Everything you're working on.",
                                color="var(--text-secondary)",
                            ),
                            align="start",
                            spacing="1",
                        ),
                        rx.spacer(),
                        rx.link(
                            rx.button(
                                rx.icon("plus", size=17),
                                "New notebook",
                                radius="full",
                                background="var(--accent-color)",
                                color="#ffffff",
                                box_shadow="0 4px 14px 0 rgba(139, 92, 246, 0.39)",
                                transition="all 0.2s ease",
                                _hover={
                                    "transform": "translateY(-1px)",
                                    "box_shadow": "0 6px 20px rgba(139, 92, 246, 0.5)",
                                },
                            ),
                            href="/create",
                            underline="none",
                        ),
                        width="100%",
                        align="center",
                    ),
                    notebook_grid(),
                    width="100%",
                    max_width="1100px",
                    margin_x="auto",
                    padding="2rem",
                    spacing="6",
                    align="stretch",
                ),
                flex="1",
                width="100%",
            ),
            width="100%",
            align="stretch",
            spacing="0",
        ),
        min_height="100vh",
        background="var(--bg-color)",
    )
