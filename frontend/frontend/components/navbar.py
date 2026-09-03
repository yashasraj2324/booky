import reflex as rx

def navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon(
                    "notebook-pen",
                    size=24,
                    stroke_width=2.5,
                    color="#f9fafb",
                ),
                rx.text(
                    "booky",
                    font_family="'Outfit', sans-serif",
                    font_size="1.4rem",
                    font_weight="800",
                    letter_spacing="-0.04em",
                    background_image="var(--accent-color)",
                    background_clip="text",
                    color="transparent",
                    style={"-webkit-background-clip": "text"},
                ),
                spacing="2",
                align="center",
            ),
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
            justify="between",
            align="center",
            width="100%",
            max_width="1200px",
            margin_x="auto",
        ),
        padding_x="2rem",
        padding_y="1rem",
        border_bottom="1px solid rgba(255, 255, 255, 0.08)",
        background="rgba(10, 10, 16, 0.65)",
        backdrop_filter="blur(16px)",
        width="100%",
        position="sticky",
        top="0",
        z_index="50",
    )
