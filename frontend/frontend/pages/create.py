import reflex as rx
from frontend.State.state import NotebookState
from frontend.components.navbar import navbar
from frontend.components.sidebar import sidebar

def create_page() -> rx.Component:
    return rx.box(
        navbar(),
        rx.hstack(
            sidebar(),
            rx.box(
                rx.vstack(
                    rx.heading(
                        "Create Notebook",
                        size="7",
                        background_image="var(--accent-color)",
                        background_clip="text",
                        color="transparent",
                        style={"-webkit-background-clip": "text"},
                        margin_bottom="1rem",
                    ),
                    rx.box(
                        rx.form(
                            rx.vstack(
                                rx.input(
                                    name="title",
                                    placeholder="Notebook Title",
                                    required=True,
                                    size="3",
                                    background="rgba(10, 10, 16, 0.5)",
                                    border_color="rgba(255, 255, 255, 0.1)",
                                    color="var(--text-primary)",
                                    _focus={"border_color": "#8B5CF6", "box_shadow": "0 0 0 1px #8B5CF6"},
                                ),
                                rx.text_area(
                                    name="description",
                                    placeholder="Description",
                                    size="3",
                                    background="rgba(10, 10, 16, 0.5)",
                                    border_color="rgba(255, 255, 255, 0.1)",
                                    color="var(--text-primary)",
                                    min_height="120px",
                                    _focus={"border_color": "#8B5CF6", "box_shadow": "0 0 0 1px #8B5CF6"},
                                ),
                                rx.button(
                                    "Create",
                                    type="submit",
                                    loading=NotebookState.creating,
                                    size="3",
                                    width="100%",
                                    background="var(--accent-color)",
                                    color="#ffffff",
                                    box_shadow="0 4px 14px 0 rgba(139, 92, 246, 0.39)",
                                    transition="all 0.2s ease",
                                    _hover={
                                        "transform": "translateY(-1px)",
                                        "box_shadow": "0 6px 20px rgba(139, 92, 246, 0.5)",
                                    },
                                ),
                                spacing="5",
                            ),
                            on_submit=NotebookState.create_notebook,
                            reset_on_submit=True,
                        ),
                        background="var(--surface-color)",
                        backdrop_filter="blur(16px)",
                        padding="2.5rem",
                        border_radius="1rem",
                        border="1px solid rgba(255, 255, 255, 0.05)",
                        box_shadow="0 10px 30px -10px rgba(0,0,0,0.5)",
                        width="100%",
                    ),
                    padding="3rem",
                    width="100%",
                    max_width="650px",
                    align="start",
                ),
                flex="1",
                width="100%",
                display="flex",
                justify_content="center",
            ),
            width="100%",
            align="stretch",
            spacing="0",
        ),
        min_height="100vh",
        background="var(--bg-color)",
    )
