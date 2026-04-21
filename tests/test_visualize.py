from eml_lab.visualize import logits_heatmap_figure


def test_logits_heatmap_figure_renders_left_and_right_axes() -> None:
    figure = logits_heatmap_figure(
        (
            {
                "cell": 0,
                "choice": "1",
                "left_probability": 0.8,
                "right_probability": 0.2,
            },
            {
                "cell": 0,
                "choice": "x",
                "left_probability": 0.2,
                "right_probability": 0.8,
            },
            {
                "cell": 1,
                "choice": "1",
                "left_probability": 0.5,
                "right_probability": 0.4,
            },
            {
                "cell": 1,
                "choice": "x",
                "left_probability": 0.3,
                "right_probability": 0.4,
            },
            {
                "cell": 1,
                "choice": "cell0",
                "left_probability": 0.2,
                "right_probability": 0.2,
            },
        )
    )

    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Left logits"
    assert figure.axes[1].get_title() == "Right logits"
    assert len(figure.axes[0].images) == 1
    assert len(figure.axes[1].images) == 1
