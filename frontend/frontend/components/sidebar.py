import reflex as rx

def sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(
                "LIBRARY",
                size="1",
                weight="bold",
                color="var(--text-secondary)",
                letter_spacing="0.08em",
            ),
            rx.button(
                rx.hstack(
                    rx.icon("library", size=17),
                    rx.text("All notebooks"),
                    spacing="2",
                ),
                variant="ghost",
                color="var(--text-primary)",
                width="100%",
                justify="start",
                _hover={
                    "background": "rgba(255, 255, 255, 0.05)",
                },
                background="rgba(255, 255, 255, 0.03)",
            ),
            spacing="3",
            width="100%",
        ),
        width="240px",
        min_height="calc(100vh - 73px)",
        padding="1.5rem",
        border_right="1px solid rgba(255, 255, 255, 0.05)",
        background="rgba(20, 20, 30, 0.4)",
    )
